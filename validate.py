import numpy as np

from train import run_once

from utils.metrics import create_resp_metric_dict, find_optimal_cutoff, create_multiclass_resp_metric_dict

def get_val_wsis_from_slide_df(cohort, new_split, slide_df):
    cohort_wsis = slide_df[slide_df.cohort==cohort].slide.values
    #print('cohort_wsis:', cohort_wsis)
    #print('split:', new_split['valid'])
    return {'valid': list(filter(lambda wsi_label: wsi_label[0] in cohort_wsis, new_split['valid']))}


def get_val_wsis_from_split(cohort, new_split):
    if cohort == 'SALZBURG':
        return {'valid': list(filter(lambda wsi: len(wsi[0]) <= 2, new_split['valid']))}
    elif cohort == 'GRAMPIAN':
        return {'valid': list(filter(lambda wsi: int(wsi[0][3]) < 2, new_split['valid']))}
    elif cohort == 'ARISTOTLE':
        return {'valid': list(filter(lambda wsi: int(wsi[0][3]) >= 2, new_split['valid']))}
    else:
        return {'valid': []}


def multiclass_validation_metrics(split, chkpt_info, epoch, arch_kwargs,
                                          loader_kwargs, args, nodes_preproc_func, val_summary_writer):
    chkpt_results, wsis = run_once(
        resp=args.resp, loss_name=args.loss, loss_weights=args.loss_weights, scale=args.scaler,
        preproc=args.preproc, temper=args.temper,
        dataset_dict=split,
        num_epochs=1,
        graph_dir=args.epi_graph_dir,
        save_dir=None,
        nodes_preproc_func=nodes_preproc_func,
        dev_mode=args.dev_mode,
        val_summary_writer=val_summary_writer,
        pretrained=chkpt_info,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs
    )

    cum_results = np.squeeze(np.array(chkpt_results))

    output_logit = []
    output_true = []

    for out in cum_results:
        output_logit.extend([out_[0] for out_ in out[:]])
        output_true.extend([out_[1] for out_ in out[:]])

    #probs = nn.functional.softmax(torch.Tensor(output_logit), dim=1).numpy()
    #pred = np.argmax(probs, axis=1)

    output_logit = np.array(output_logit, dtype=np.float16)
    output_true = np.array(output_true)

    print(args.resp[0])
    metric_dict = create_multiclass_resp_metric_dict(args.resp[0], output_true, output_logit, epoch)

    return metric_dict

# Check validation metrics on different cohorts
def validation_metrics(split, chkpt_info, epoch, arch_kwargs,
                       loader_kwargs, args, nodes_preproc_func, val_summary_writer, thresholds=None):
    chkpt_results, wsis = run_once(
        resp=args.resp, loss_name=args.loss, loss_weights=args.loss_weights, scale=args.scaler,
        preproc=args.preproc, temper=args.temper,
        dataset_dict=split,
        num_epochs=1,
        graph_dir=args.epi_graph_dir,
        save_dir=None,
        nodes_preproc_func=nodes_preproc_func,
        dev_mode=args.dev_mode,
        val_summary_writer=val_summary_writer,
        pretrained=chkpt_info,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs
    )

    # * re-calibrate logit to probabilities
    chkpt_results = np.array(chkpt_results)
    chkpt_results = np.squeeze(chkpt_results)

    # one fold only
    cum_results = chkpt_results
    cum_results = np.array(cum_results)
    cum_results = np.squeeze(cum_results)

    output_1_logit, output_1_true = [], []
    output_2_logit, output_2_true = [], []
    node_output_logit, node_output_true = [], []

    if 'cohort_cls' in args.resp:
        epi_idx = 3
    else:
        epi_idx = 2

    for out in cum_results:
        output_1_logit.append(out[0][0])
        output_1_true.append(out[0][1])

        output_2_logit.append(out[1][0])
        output_2_true.append(out[1][1])

        node_output_logit.extend([out_[0] for out_ in out[epi_idx:]])
        node_output_true.extend([out_[1] for out_ in out[epi_idx:]])

    output_1_logit = np.array(output_1_logit)
    output_1_true = np.array(output_1_true)
    output_2_logit = np.array(output_2_logit)
    output_2_true = np.array(output_2_true)
    node_output_logit = np.array(node_output_logit)
    node_output_true = np.array(node_output_true)

    print('Without thresholding')
    metric_dict = {}
    print(args.resp[0])
    metric_dict.update(create_resp_metric_dict(args.resp[0], output_1_true, output_1_logit, epoch))
    print(args.resp[1])
    metric_dict.update(create_resp_metric_dict(args.resp[1], output_2_true, output_2_logit, epoch))
    print(args.resp[-1])
    metric_dict.update(create_resp_metric_dict(args.resp[-1], node_output_true, node_output_logit, epoch))

    if thresholds is not None:
        print('Using thresholds from validation')
        threshold_0, threshold_1, threshold_2 = thresholds
    else:
        print('Using thresholding from joint cohorts')
        threshold_0 = find_optimal_cutoff(output_1_true, output_1_logit)[0]
        threshold_1 = find_optimal_cutoff(output_2_true, output_2_logit)[0]
        threshold_2 = find_optimal_cutoff(node_output_true, node_output_logit)[0]

    print(f'Thresholds: {threshold_0}, {threshold_1}, {threshold_2}')

    print(args.resp[0])
    resp_0_mets = create_resp_metric_dict(args.resp[0], output_1_true, output_1_logit, epoch, cutoff=threshold_0)
    resp_0_mets = {'threshold-' + k: v for k, v in resp_0_mets.items() if not k == 'best_epoch'}
    resp_0_mets[f'{args.resp[0]}-threshold'] = threshold_0
    metric_dict.update(resp_0_mets)

    print(args.resp[1])
    resp_1_mets = create_resp_metric_dict(args.resp[1], output_2_true, output_2_logit, epoch, cutoff=threshold_1)
    resp_1_mets = {'threshold-' + k: v for k, v in resp_1_mets.items() if not k == 'best_epoch'}
    resp_1_mets[f'{args.resp[1]}-threshold'] = threshold_1
    metric_dict.update(resp_1_mets)

    print(args.resp[-1])
    resp_2_mets = create_resp_metric_dict(args.resp[-1], node_output_true, node_output_logit, epoch, cutoff=threshold_2)
    resp_2_mets = {'threshold-' + k: v for k, v in resp_2_mets.items() if not k == 'best_epoch'}
    resp_2_mets[f'{args.resp[-1]}-threshold'] = threshold_2
    metric_dict.update(resp_2_mets)

    return metric_dict