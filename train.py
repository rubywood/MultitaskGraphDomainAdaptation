import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import average_precision_score as auprc_scorer#
from sklearn.linear_model import LogisticRegression
import os

# Local imports
from utils.model import SlideGraphArch, slidegraph_loss_function, SlideGraphArchMLPv2, GNNMLPv3, GNNMLPv4
from utils.data import StratifiedSampler, SlideGraphEpiDataset, collate_fn_pad
from utils.helper import ScalarMovingAverage, create_pbar
from utils.utils import load_json, save_as_json


def load_model(version, responses, arch_kwargs):
    if version == 4:
        model = GNNMLPv4(responses=responses, **arch_kwargs)
    elif version == 3:
        model = GNNMLPv3(responses=responses, **arch_kwargs)
    elif version == 2:
        model = SlideGraphArchMLPv2(responses=responses, **arch_kwargs)
    else:
        model = SlideGraphArch(responses=responses, **arch_kwargs)
    return model


def run_once(
    resp,
    loss_name,
    loss_weights,
    scale, #=args.scaler,
    preproc,
    temper,
    dataset_dict,
    num_epochs,
    graph_dir,
    save_dir,
    nodes_preproc_func=None,
    dev_mode=False,
    train_summary_writer=None,
    val_summary_writer=None,
    pretrained=None,
    loader_kwargs={},
    arch_kwargs={},
    optim_kwargs={},
):
    """Running the inference or training loop once.

    The actual running mode is defined via the code name of the dataset
    within `dataset_dict`. Here, `train` is specifically preserved for
    the dataset used for training. `.*infer-valid.*` and `.*infer-train*`
    are reserved for datasets containing the corresponding labels.
    Otherwise, the dataset is assumed to be for the inference run.

    """
    model = load_model(arch_kwargs['mlp_version'], resp, arch_kwargs)

    if pretrained is not None:
        model.load(*pretrained)
    model = model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    if loss_name == 'bce':
        criterion = nn.BCELoss().cuda()
        multiclass_criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = None

    # Create the graph dataset holder for each subset info then
    # pipe them through torch/torch geometric specific loader
    # for loading in multi-thread.
    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        _loader_kwargs['batch_size'] = 1 # RW: added because otherwise batch size 16 fails for validation
        batch_sampler = None
        if subset_name == "train":
            _loader_kwargs = {}
            batch_sampler = StratifiedSampler(
                labels=[v[1] for v in subset], batch_size=loader_kwargs["batch_size"]
            )  # batch_size must be less than number of labels to allow for >0 data splits/batches

        if len(os.listdir(graph_dir)) == 2:
            if 'train' in subset_name:
                graph_dir = os.path.join(graph_dir, 'Train')
            else:
                graph_dir = os.path.join(graph_dir, 'Validation')

        if preproc:
            ds = SlideGraphEpiDataset(subset, graph_dir=graph_dir, mode=subset_name, preproc=nodes_preproc_func)
        else:
            ds = SlideGraphEpiDataset(subset, graph_dir=graph_dir, mode=subset_name, preproc=None)
        loader_dict[subset_name] = torch.utils.data.DataLoader( # changed from geometric to normal dataloader
            ds,
            collate_fn=collate_fn_pad,
            batch_sampler=batch_sampler,
            drop_last=subset_name == "train" and batch_sampler is None,
            shuffle=subset_name == "train" and batch_sampler is None,
            **_loader_kwargs,
        )
        print(f'Loader for {subset_name} is length {len(loader_dict[subset_name])}')

    for epoch in range(num_epochs):
        #logging.info(f"EPOCH {epoch:03d}")
        for loader_name, loader in loader_dict.items():
            # * EPOCH START
            step_output, train_step_output = [], []
            step_wsis = []
            ema = ScalarMovingAverage()
            pbar = create_pbar(loader_name, len(loader))
            print('Loader length:', len(loader))
            epoch_loss = []
            for step, batch_data in enumerate(loader):
                # * STEP COMPLETE CALLBACKS
                if 'train' in loader_name: #if loader_name == "train":
                    output = model.train_batch(model, batch_data[0], resp, loss_name, loss_weights, optimizer, criterion,
                                               temper=temper)
                    #print('Output[0]:', output[0])
                    # output = [loss, output_dict, wsi_labels]
                    if 'infer' in loader_name:  # 'infer-train' for LR
                        train_wsi_labels = output[2]
                        train_output_dict = output[1]
                        train_output_list = []
                        for i in range(len(resp)):
                            if 'epi' in resp[i]:
                                output_i = train_output_dict[resp[i]][1].squeeze().detach().cpu().numpy()  # node activations
                                train_output_list.extend(output_i)
                            elif resp[i] != 'cohort_cls':  # don't add cohort prediction
                                output_i = train_output_dict[resp[i]][0].squeeze().detach().cpu().item()  # scalars
                                train_output_list.append(output_i)
                        train_wsi_labels = np.squeeze(train_wsi_labels)
                        if 'cohort_cls' in resp:
                            cohort_cls_idx = resp.index('cohort_cls')
                            # Remove cohort cls label
                            train_wsi_labels = np.delete(train_wsi_labels, cohort_cls_idx)
                        # keep responses separate and zip for each WSI
                        if "label" in batch_data[0]:
                            train_output = list(zip(train_output_list, train_wsi_labels))
                        else:
                            train_output = train_output_list
                        train_step_output.append(train_output)
                    else:
                        ema({"loss": output[0]})
                        epoch_loss.append(output[0])
                        pbar.postfix[1]["step"] = output[0]
                        pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else: # don't want to run for infer-train, use train output for LR
                    output_dict, labels = model.infer_batch(model, batch_data[0])
                    #batch_size = batch_data[0]["graph"].num_graphs
                    
                    output_list = []
                    for i in range(len(resp)):
                        if resp[i] in ['CMS', 'CMS_matching'] or 'epi' in resp[i]:
                        #if any(v in resp[i] for v in ['epi', 'CMS']): # didn't work for binary CMS4
                            output_i = output_dict[resp[i]][1].cpu().squeeze().numpy() # node activations
                            output_list.extend(output_i)
                        elif resp[i]!='cohort_cls': # don't add cohort prediction
                            output_i = output_dict[resp[i]][0].cpu().squeeze().item() # scalars
                            output_list.append(output_i)

                    labels = np.squeeze(labels)
                    if 'cohort_cls' in resp:
                        cohort_cls_idx = resp.index('cohort_cls')
                        # Remove cohort cls label
                        #print('Size of labels:', labels.shape)
                        labels = np.delete(labels, cohort_cls_idx)
                    # keep responses separate and zip for each WSI
                    if "label" in batch_data[0]:
                        output = list(zip(output_list, labels))
                    else:
                        output = output_list

                    step_output.append(output)
                    step_wsis.extend(batch_data[1])
                pbar.update()
                del step, batch_data, output
            pbar.close()
            # * EPOCH COMPLETE

            # Callbacks to process output
            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
                if train_summary_writer:
                    train_summary_writer.add_scalar(f'Loss', logging_dict['train-EMA-loss'], epoch)
            elif "infer" in loader_name and any(
                v in loader_name for v in ["train", "valid"]
            ):
                # Expand the list of N dataset size x H heads
                # back to a list of H Head each with N samples.
                ####################################
                loss = 0

                if 'train' in loader_name:
                    for i in range(len(resp)):
                        train_output_logit, train_output_true = [], []
                        for out in train_step_output:
                            if any(v in resp[i] for v in ['epi', 'CMS']):
                                train_output_logit.extend([out_[0] for out_ in out[i:]])
                                train_output_true.extend([out_[1] for out_ in out[i:]])
                            else:
                                train_output_logit.append(out[i][0])
                                train_output_true.append(out[i][1])

                        train_output_logit = np.array(train_output_logit)
                        train_output_true = np.array(train_output_true)

                        if scale:
                            scaler = LogisticRegression()
                            scaler.fit(np.array(train_output_logit, ndmin=2).T, train_output_true)
                            model.aux_model[f"scaler_{resp[i]}"] = scaler

                else: #valid
                    for i in range(len(resp)):
                        output_logit, output_true = [], []
                        for out in step_output:
                            if any(v in resp[i] for v in ['epi', 'CMS']):
                                output_logit.extend([out_[0] for out_ in out[i:]])
                                output_true.extend([out_[1] for out_ in out[i:]])
                            else:
                                output_logit.append(out[i][0])
                                output_true.append(out[i][1])

                        output_logit = np.array(output_logit, dtype=np.float16)
                        output_true = np.array(output_true)

                        if scale:
                            # predict using trained LR on validation predictions
                            scaler = model.aux_model[f"scaler_{resp[i]}"]
                            output_logit = scaler.predict_proba(np.array(output_logit, ndmin=2).T)[:, 0]
                            # between 0 and 1

                        #if not ("train" in loader_name):
                        if 'epi' in resp[i]:
                            # Epithelial loss
                            if loss_name == 'slidegraph':
                                #wsi_labels_ = wsi_labels[:,None]
                                wsi_labels = output_true.reshape(len(output_true),1)

                                #wsi_output = wsi_output[:,None]
                                wsi_output = output_logit.reshape(len(output_logit),1)

                                # have to split for memory capacity reasons
                                n_splits = 5
                                wsi_output_n = np.array_split(wsi_output, n_splits)
                                wsi_labels_n = np.array_split(wsi_labels, n_splits)

                                diff = np.array([])
                                for i in range(n_splits):
                                    wsi_output_i = wsi_output_n[i]
                                    wsi_labels_i = wsi_labels_n[i]

                                    wsi_output_ = wsi_output_i - wsi_output_i.T
                                    wsi_labels_ = wsi_labels_i - wsi_labels_i.T
                                    del wsi_output_i, wsi_labels_i

                                    diff = np.append(diff, (wsi_output_[wsi_labels_>0]))
                                    del wsi_output_, wsi_labels_

                                resp_loss = torch.mean(F.relu(1.0 - torch.Tensor(diff)))
                            elif loss_name == 'bce':
                                resp_loss = criterion(torch.FloatTensor(output_logit).cuda(),
                                                      torch.FloatTensor(output_true).cuda())
                            else:
                                raise Exception('loss not defined')

                        elif resp[i] != 'cohort_cls':
                            if resp[i] in ['CMS_matching', 'CMS']:
                                resp_loss = multiclass_criterion(torch.Tensor(output_logit).cuda(),
                                                            torch.Tensor(output_true).type(torch.LongTensor).cuda())
                            elif loss_name == 'slidegraph':
                                resp_loss = slidegraph_loss_function(output_true, output_logit[:, None])
                            elif loss_name == 'bce':
                                resp_loss = criterion(torch.Tensor(output_logit).cuda(),
                                                      torch.Tensor(output_true).cuda())
                            else:
                                raise Exception('loss not defined')

                        loss += loss_weights[i] * resp_loss


                        if resp[i] != 'cohort_cls':
                            try:
                                if resp[i] in ['CMS_matching', 'CMS']:
                                    output_logit = nn.functional.softmax(torch.Tensor(output_logit), dim=1).numpy()

                                    #print('output_true.shape', output_true.shape)
                                    #print('output_logit.shape', output_logit.shape)
                                    #print('output_logit:', output_logit)
                                    #print('output_true:', output_true)
                                    # pass (n_samples, n_classes) of probability estimates

                                    auc = auroc_scorer(output_true, output_logit,
                                                       average='macro', multi_class='ovr')
                                    print('auc:', auc)
                                else:
                                    auc = auroc_scorer(output_true, output_logit)
                                    auprc = auprc_scorer(output_true, output_logit)
                                    logging_dict[f"{resp[i]}-{loader_name}-auprc"] = auprc
                                logging_dict[f"{resp[i]}-{loader_name}-auroc"] = auc
                            except ValueError as e:
                                print(f"Couldn't calculate metrics due to error: {e}")

                            logging_dict[f"{resp[i]}-{loader_name}-raw-logit"] = output_logit
                            logging_dict[f"{resp[i]}-{loader_name}-raw-true"] = output_true

                            if val_summary_writer is not None:
                                val_summary_writer.add_scalar(f'{resp[i]}-loss', resp_loss.item(), epoch)
                                val_summary_writer.add_scalar(f'{resp[i]}-AUC', auc, epoch)
                                if not resp[i] in ['CMS_matching', 'CMS']:
                                    val_summary_writer.add_scalar(f'{resp[i]}-AUPRC', auprc, epoch)
                        del output_logit, output_true

                        #if val_summary_writer is not None:
                        #    if resp[i] != 'cohort_cls':
                        #        val_summary_writer.add_scalar(f'{resp[i]}-loss', resp_loss.item(), epoch)
                        #
                        #        try:
                        #            if not resp[i] in ['CMS_matching', 'CMS']:
                        #                val_summary_writer.add_scalar(f'{resp[i]}-AUPRC', auprc, epoch)
                        #            val_summary_writer.add_scalar(f'{resp[i]}-AUC', auc, epoch)
                        #        except ValueError as e:
                        #            print(f"Couldn't calculate metrics due to error: {e}")
                
                if not ("train" in loader_name):
                    if temper is not None:
                        loss = loss * temper
                    loss = loss.detach().cpu().numpy()
                    logging_dict[f"{loader_name}-loss"] = loss.item()
                    if val_summary_writer is not None:
                        val_summary_writer.add_scalar(f'Loss', loss.item(), epoch)
                        

            # Callbacks for logging and saving
            #for val_name, val in logging_dict.items():
            #    if "raw" not in val_name:
            #        logging.info(f"{val_name}: {val}")
            if "train" not in loader_dict:
                continue

            if not dev_mode:
                # Track the statistics
                new_stats = {}
                if os.path.exists(f"{save_dir}/stats.json"):
                    old_stats = load_json(f"{save_dir}/stats.json")
                    # Save a backup first
                    save_as_json(old_stats, f"{save_dir}/stats.old.json", exist_ok=True)
                    new_stats = copy.deepcopy(old_stats)
                    new_stats = {int(k): v for k, v in new_stats.items()}

                old_epoch_stats = {}
                if epoch in new_stats:
                    old_epoch_stats = new_stats[epoch]
                old_epoch_stats.update(logging_dict)
                new_stats[epoch] = old_epoch_stats
                save_as_json(new_stats, f"{save_dir}/stats.json", exist_ok=True) # RW: changed exist_ok to True
                del new_stats, old_epoch_stats

                # Save the pytorch model
                model.save(
                    f"{save_dir}/epoch={epoch:03d}.weights.pth",
                    f"{save_dir}/epoch={epoch:03d}.aux.dat",
                )

    return step_output, step_wsis