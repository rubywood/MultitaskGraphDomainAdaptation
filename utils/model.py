import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.nn import (
    EdgeConv,
    GINConv,
    GATConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    MLP
)
#from torch_geometric.nn.aggr import MLPAggregation

import numpy as np
import joblib
import pathlib
from collections import OrderedDict
from typing import Tuple

from .utils import load_json

from typing import Optional
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation

'''Builds on SlideGraph https://github.com/wenqi006/SlideGraph'''

# Correcting torch_geometric.nn.aggr.mlp
class MLPAggregation(Aggregation):
    r"""Performs MLP aggregation in which the elements to aggregate are
    flattened into a single vectorial representation, and are then processed by
    a Multi-Layer Perceptron (MLP), as described in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    .. note::

        :class:`GRUAggregation` requires sorted indices :obj:`index` as input.
        Specifically, if you use this aggregation as part of
        :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
        or by calling :meth:`torch_geometric.data.Data.sort`.

    .. warning::

        :class:`MLPAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        max_num_elements (int): The maximum number of elements to aggregate per
            group.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.models.MLP`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 max_num_elements: int, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_elements = max_num_elements

        self.mlp = MLP(in_channels=in_channels * max_num_elements,
                       #hidden_channels=int((in_channels * max_num_elements) / 2),
                       out_channels=out_channels, num_layers=1, **kwargs)
        # [1, 32] * [16, 1]

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=self.max_num_elements)
        #print('After dense batch in MLP:', x.shape) #After dense batch in MLP: torch.Size([1, 1, 32])
        x = x.view(-1, x.size(1) * x.size(2))
        return self.mlp(x, index, dim_size)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'max_num_elements={self.max_num_elements})')

# TODO: add pooling and dropout between GNN layers - only relevant to node predictions
# linear, pooling, dropout -> GNN convs with (linear,) pooling, dropout -> MLP with 1 hidden layer -> pooling -> temper

# Simple GNN: linear -> GNN convs -> (split into branches) MLP v4 with 3 layers -> pooling -> dropout -> temper
class GNNMLPv4(nn.Module):
    def __init__(
            self,
            responses,
            dim_features,
            dim_target,
            layers=[6, 6],
            pooling="max",
            dropout=0.0,
            conv="GINConv",
            gembed=False,
            scale=False,
            temper=None,
            use_mlp=True,
            mlp_dropout=0.1,
            label_dim=[1,1,1], # TODO: BINARY
            **kwargs
    ):
        super().__init__()
        self.responses = responses
        self.dropout = dropout
        self.embeddings_dim = layers
        self.num_layers = len(self.embeddings_dim)
        self.convs = []
        self.linears = []
        self.branch_linears = []
        self.mlp_heads = []
        self.mlpv4 = []
        self.pooling = {
            "max": global_max_pool,
            "mean": global_mean_pool,
            "add": global_add_pool,
        }[pooling]
        # If True then learn a graph embedding for final classification
        # (classify pooled node features), otherwise pool node decision scores.
        self.gembed = gembed
        self.temper = temper
        self.label_dim = label_dim # TODO: BINARY
        assert len(self.label_dim) == len(self.responses), "Binary list and responses must be equal length" # TODO: BINARY

        conv_dict = {"GINConv": [GINConv, 1], "EdgeConv": [EdgeConv, 2], "GATConv": [GATConv, 1]}  # changed from 1 to 2
        if conv not in conv_dict:
            raise ValueError(f'Not support `conv="{conv}".')

        def create_linear(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims), BatchNorm1d(out_dims), ReLU()
            )

        input_emb_dim = dim_features
        out_emb_dim = self.embeddings_dim[0]
        self.first_h = create_linear(input_emb_dim, out_emb_dim)

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embeddings_dim[1:]:
            ConvClass, alpha = conv_dict[conv]
            if conv == 'GATConv':
                self.convs.append(ConvClass(in_channels=alpha * input_emb_dim,
                                            out_channels=out_emb_dim, **kwargs))
            else:
                subnet = create_linear(alpha * input_emb_dim, out_emb_dim)

                self.convs.append(ConvClass(subnet, **kwargs))
            input_emb_dim = out_emb_dim

        # add MLP conv for final layer, x3 for each response

        for i in range(len(responses)):
            # 3 layers, e.g. 16 -> 8 -> 8 -> 1
            self.mlpv4.append(MLP(in_channels=int(self.embeddings_dim[-2]),
                                  out_channels=label_dim[i],
                                  hidden_channels=int(self.embeddings_dim[-1]),
                                  num_layers=3,
                                  norm="batch_norm",
                                  act="relu",
                                  bias=True,
                                  dropout=mlp_dropout))

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1) # TODO: BINARY
        self.scale = scale

        self.convs = torch.nn.ModuleList(self.convs)
        self.mlpv4 = torch.nn.ModuleList(self.mlpv4)  # this puts it onto cuda

        # Auxilary holder for external model, these are saved separately from torch.save
        # as they can be sklearn model etc.
        self.aux_model = {}

    def save(self, path, aux_path=None):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        if aux_path is not None:
            joblib.dump(self.aux_model, aux_path)

    def load(self, path, aux_path=None):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        if aux_path is not None:
            self.aux_model = joblib.load(aux_path)

    def forward(self, data):

        edge_index = data.edge_index
        if edge_index.dtype == torch.float64:
            edge_index = data.edge_index.long()
        feature, batch = data.x, data.batch

        # wsi_prediction = 0
        pooling = self.pooling
        # node_prediction = 0
        out_dict = OrderedDict()

        for layer in range(self.num_layers):
            if layer == 0:
                feature = self.first_h(feature)
            elif layer == self.num_layers - 1:
                # Final layer, branch for each response in output
                # branches_feature = self.convs[layer - 1](feature, edge_index)
                # print('Features in final layer:', branches_feature.shape)

                # Features in middle layers: torch.Size([406164, 32])
                # Node_prediction_sub in middle layers: torch.Size([406164, 1])
                # Features in final layer: torch.Size([406164, 16])
                # Node prediction in final layer: torch.Size([406164, 1])

                for i in range(len(self.responses)):
                    # _wsi_prediction = wsi_prediction.clone()
                    # _node_prediction = node_prediction.clone()
                    # print('Node prediction in final layer:', _node_prediction.shape) #[n, 1]

                    _node_prediction = self.mlpv4[i](feature)  # MLP includes final layer as 1
                    # print('Node prediction shape:', _node_prediction.shape)
                    node_pooled = pooling(_node_prediction, batch)
                    _wsi_prediction = F.dropout(
                        node_pooled, p=self.dropout, training=self.training
                    )

                    if not self.scale:
                        if self.temper is not None:
                            _wsi_prediction = _wsi_prediction / self.temper
                            _node_prediction = _node_prediction / self.temper
                        if self.label_dim[i] == 1: #TODO: BINARY
                            _wsi_prediction = self.sig(_wsi_prediction)
                            _node_prediction = self.sig(_node_prediction)
                        # if want probabilities, use softmax. For CE loss, use raw logits.
                    out_dict[self.responses[i]] = [_wsi_prediction, _node_prediction]
            else:
                # All other layers, with GinConv then linear, pooling and dropout
                feature = self.convs[layer - 1](feature, edge_index)
        return out_dict

    # output dict with {'response_cr_nocr': [wsi_pred, node_pred], 'CMS4': [wsi_pred, node_pred], etc.}

    # Run one single step
    @staticmethod
    def train_batch(model, batch_data, responses, loss_name, loss_weights, optimizer: torch.optim.Optimizer,
                    criterion=None, temper=None):
        wsi_graphs = batch_data["graph"].to("cuda")
        wsi_labels = batch_data["label"].to("cuda")  # both labels for both responses like [0, 1]

        # remove padding from labels
        # print('WSI labels length before removing padding:', len(wsi_labels))
        orig_lengths = batch_data["length"]  # .to("cuda")
        # print('Original labels length:', len(orig_lengths))
        wsi_labels = [wsi_labels[i, :orig_lengths[i]] for i in range(len(orig_lengths))]
        # print('WSI labels length:', len(wsi_labels))

        # WSI GRAPHS: DataBatch(x=[589, 384], edge_index=[2, 3366], coords=[589, 2], batch=[589], ptr=[4])
        # WSI LABELS: tensor([1., 1., 1.], device='cuda:0', dtype=torch.float64)

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Not an RNN so does not accumulate
        model.train()
        optimizer.zero_grad()

        output_dict = model(wsi_graphs)

        multiclass_criterion = nn.CrossEntropyLoss().cuda()

        loss = 0
        for i in range(len(responses)): #range(len(responses[:2])):  # works for less than 2 responses also # TODO: BINARY
            if responses[i] == 'cohort_cls':
                continue
            elif 'epithelium' in responses[i]: # TODO: BINARY
                continue # TODO: BINARY
            labels = torch.stack([wsi_labels[j][i] for j in range(len(wsi_labels))])
            output = output_dict[responses[i]][0]

            #print('Labels:', labels.squeeze())
            #print('Output:', output.squeeze())

            if responses[i] in ['CMS_matching', 'CMS']: # TODO: BINARY
                loss += loss_weights[i] * multiclass_criterion(output.squeeze(),  # TODO: BINARY
                                                               labels.squeeze().type(torch.LongTensor).cuda())
            elif loss_name == 'slidegraph':
                labels_ = labels[:, None]
                labels_ = labels_ - labels_.T
                output_ = output - output.T
                diff = output_[labels_ > 0]
                resp_loss = torch.mean(F.relu(1.0 - diff))
                loss += loss_weights[i] * resp_loss
            elif loss_name == 'bce':
                # node_output_ = node_output.squeeze().cuda()
                # labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
                loss += loss_weights[i] * criterion(output.squeeze(), labels.squeeze().type(torch.FloatTensor).cuda())

        if 'cohort_cls' in responses:
            cohort_idx = responses.index('cohort_cls')
            # if responses[2] == 'cohort_cls':
            labels = torch.stack([wsi_labels[j][cohort_idx] for j in range(len(wsi_labels))])
            output = output_dict[responses[cohort_idx]][0]

            # Negative loss for cohort - don't want to be able to predict. Changed to dividing later.
            cohort_loss = loss_weights[cohort_idx] * multiclass_criterion(output.squeeze(),
                                                                      labels.squeeze().type(torch.FloatTensor).cuda())
            # loss -= cohort_loss

            print('Cohort training loss:', cohort_loss)

        if any('epithelium' in resp for resp in responses):
            epi_label_idx = [idx for idx, s in enumerate(responses) if 'epi' in s][0]
            # For epithelial response
            node_output = output_dict[responses[-1]][1]
            labels = torch.cat([wsi_labels[j][epi_label_idx:] for j in range(len(wsi_labels))])  # cat flattens lists

            if loss_name == 'slidegraph':
                labels = labels.reshape(len(labels), 1)
                # wsi_output = flat_logit.reshape(len(flat_logit),1)

                n_splits = 10
                node_output_n = np.array_split(node_output, n_splits)
                labels_n = np.array_split(labels, n_splits)

                diff = torch.Tensor([]).cuda()
                for i in range(n_splits):
                    node_output_i = node_output_n[i]
                    labels_i = labels_n[i]

                    node_output_ = node_output_i - node_output_i.T
                    labels_ = labels_i - labels_i.T
                    del node_output_i, labels_i

                    diff = torch.cat((diff, (node_output_[labels_ > 0])))
                    del node_output_, labels_

                loss += loss_weights[-1] * torch.mean(F.relu(1.0 - diff))

            elif loss_name == 'bce':
                node_output_ = node_output.squeeze().cuda()
                labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
                loss += loss_weights[-1] * criterion(node_output_, labels_)
            else:
                raise Exception('loss not defined')

        # if responses[2] == 'cohort_cls':
        if 'cohort_cls' in responses:
            loss = loss / cohort_loss
            # loss += cohort_loss

        if temper is not None:
            loss = loss * temper

        # Backprop and update
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = [wsi_label.cpu().numpy() for wsi_label in wsi_labels]
        return [loss, output_dict, wsi_labels]

    # Run one inference step
    @staticmethod
    def infer_batch(model, batch_data):
        wsi_graphs = batch_data["graph"].to("cuda")

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            output_dict = model(wsi_graphs)  # contains wsi and node predictions

        # Output should be a single tensor or scalar
        if "label" in batch_data:
            wsi_labels = batch_data["label"]
            wsi_labels = wsi_labels.cpu().numpy()
            return output_dict, wsi_labels
        return output_dict, _
        # return [output_dict]

# Simple GNN: linear -> GNN convs -> (split into branches) MLP v3 with 2 layers -> pooling -> dropout -> temper
# removed cumulative summing of node predictions, just forward features with MLP on features at end
class GNNMLPv3(nn.Module):
    def __init__(
            self,
            responses,
            dim_features,
            dim_target,
            layers=[6, 6],
            pooling="max",
            dropout=0.0,
            conv="GINConv",
            gembed=False,
            scale=False,
            temper=None,
            use_mlp=True,
            mlp_dropout=0.1,
            label_dim=[1, 1, 1],  # TODO: BINARY
            **kwargs
    ):
        super().__init__()
        self.responses = responses
        self.dropout = dropout
        self.embeddings_dim = layers
        self.num_layers = len(self.embeddings_dim)
        self.convs = []
        self.linears = []
        self.branch_linears = []
        self.mlp_heads = []
        self.mlpv3 = []
        self.pooling = {
            "max": global_max_pool,
            "mean": global_mean_pool,
            "add": global_add_pool,
        }[pooling]
        # If True then learn a graph embedding for final classification
        # (classify pooled node features), otherwise pool node decision scores.
        self.gembed = gembed
        self.temper = temper
        self.use_mlp = use_mlp
        self.label_dim = label_dim  # TODO: BINARY
        assert len(self.label_dim) == len(self.responses), \
            "Binary list and responses must be equal length"  # TODO: BINARY

        conv_dict = {"GINConv": [GINConv, 1], "EdgeConv": [EdgeConv, 2], "GATConv": [GATConv, 1]}  # changed from 1 to 2
        if conv not in conv_dict:
            raise ValueError(f'Not support `conv="{conv}".')

        def create_linear(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims), BatchNorm1d(out_dims), ReLU()
            )

        input_emb_dim = dim_features
        out_emb_dim = self.embeddings_dim[0]
        self.first_h = create_linear(input_emb_dim, out_emb_dim)
        self.linears.append(Linear(out_emb_dim, dim_target))

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embeddings_dim[1:]:
            ConvClass, alpha = conv_dict[conv]
            if conv == 'GATConv':
                self.convs.append(ConvClass(in_channels=alpha * input_emb_dim,
                                            out_channels=out_emb_dim, **kwargs))
            else:
                subnet = create_linear(alpha * input_emb_dim, out_emb_dim)

                self.convs.append(ConvClass(subnet, **kwargs))
            self.linears.append(Linear(out_emb_dim, dim_target))
            input_emb_dim = out_emb_dim

        # TODO: add MLP conv for final layer, x3 for each response

        for i in range(len(responses)):
            # Output node prediction so don't go through GNN again
            # 16 -> 8 -> 1 with norm and activation
            self.mlpv3.append(MLP(in_channels=int(self.embeddings_dim[-2]),
                                  out_channels=label_dim[i],
                                  hidden_channels=int(self.embeddings_dim[-1]),
                                  num_layers=2,
                                  norm="batch_norm",
                                  act="relu",
                                  bias=True,
                                  dropout=mlp_dropout))

        self.sig = nn.Sigmoid()
        self.scale = scale

        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)
        self.mlpv3 = torch.nn.ModuleList(self.mlpv3)  # this puts it onto cuda
        self.dim_target = dim_target

        # Auxilary holder for external model, these are saved separately from torch.save
        # as they can be sklearn model etc.
        self.aux_model = {}

    def save(self, path, aux_path=None):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        if aux_path is not None:
            joblib.dump(self.aux_model, aux_path)

    def load(self, path, aux_path=None):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        if aux_path is not None:
            self.aux_model = joblib.load(aux_path)

    def forward(self, data):

        edge_index = data.edge_index
        if edge_index.dtype == torch.float64:
            edge_index = data.edge_index.long()
        feature, batch = data.x, data.batch

        #wsi_prediction = 0
        pooling = self.pooling
        #node_prediction = 0
        out_dict = OrderedDict()

        
        for layer in range(self.num_layers):
            if layer == 0:
                feature = self.first_h(feature)
            elif layer == self.num_layers - 1:
                for i in range(len(self.responses)):
                    #_wsi_prediction = wsi_prediction.clone()
                    #_node_prediction = node_prediction.clone()
                    #print('Node prediction in final layer:', _node_prediction.shape) #[n, 1]
                              
                    _node_prediction = self.mlpv3[i](feature) # MLP includes final layer as 1
                    #print('Node prediction shape:', _node_prediction.shape)
                    node_pooled = pooling(_node_prediction, batch)
                    _wsi_prediction = F.dropout(
                        node_pooled, p=self.dropout, training=self.training
                    )
                    #print('WSI prediction shape:', _wsi_prediction.shape)

                    if not self.scale:
                        if self.temper is not None:
                            _wsi_prediction = _wsi_prediction / self.temper
                            _node_prediction = _node_prediction / self.temper
                        #if self.use_mlp:
                        #    _wsi_prediction = self.mlp_heads[i](_wsi_prediction)
                        #    _node_prediction = self.mlp_heads[i](_node_prediction)
                        #else:
                        if self.label_dim[i] == 1: #TODO: BINARY
                            _wsi_prediction = self.sig(_wsi_prediction)
                            _node_prediction = self.sig(_node_prediction)
                    out_dict[self.responses[i]] = [_wsi_prediction, _node_prediction]

                # Returning features as well
                out_dict['features'] = feature
            else:
                if layer == self.num_layers - 2:
                    # Returning features as well
                    out_dict['featuresfc1'] = feature
                # All other layers, with GinConv then linear, pooling and dropout
                feature = self.convs[layer - 1](feature, edge_index)

                #print('Features in middle layers:', feature.shape)
                #if not self.gembed:
                #    node_prediction_sub = self.linears[layer](feature)
                #    #print('Node prediction sub in middle layers:', node_prediction_sub.shape)
                #    node_prediction += node_prediction_sub
                #    node_pooled = pooling(node_prediction_sub, batch)
                #    wsi_prediction_sub = F.dropout(
                #        node_pooled, p=self.dropout, training=self.training
                #    )
                #else:
                #    node_pooled = pooling(feature, batch)
                #    node_prediction_sub = self.linears[layer](node_pooled)
                #    wsi_prediction_sub = F.dropout(
                #        node_prediction_sub, p=self.dropout, training=self.training
                #    )
                #wsi_prediction += wsi_prediction_sub
        return out_dict

    # output dict with {'response_cr_nocr': [wsi_pred, node_pred], 'CMS4': [wsi_pred, node_pred], etc.}

    # Run one single step
    @staticmethod
    def train_batch(model, batch_data, responses, loss_name, loss_weights, optimizer: torch.optim.Optimizer,
                    criterion=None, temper=None):
        wsi_graphs = batch_data["graph"].to("cuda")
        wsi_labels = batch_data["label"].to("cuda")  # both labels for both responses like [0, 1]

        # remove padding from labels
        #print('WSI labels length before removing padding:', len(wsi_labels))
        orig_lengths = batch_data["length"]  # .to("cuda")
        #print('Original labels length:', len(orig_lengths))
        wsi_labels = [wsi_labels[i, :orig_lengths[i]] for i in range(len(orig_lengths))]
        #print('WSI labels length:', len(wsi_labels))

        # WSI GRAPHS: DataBatch(x=[589, 384], edge_index=[2, 3366], coords=[589, 2], batch=[589], ptr=[4])
        # WSI LABELS: tensor([1., 1., 1.], device='cuda:0', dtype=torch.float64)

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Not an RNN so does not accumulate
        model.train()
        optimizer.zero_grad()

        output_dict = model(wsi_graphs)

        multiclass_criterion = nn.CrossEntropyLoss().cuda()

        loss = 0
        for i in range(len(responses)): # works for less than 2 responses also
            if responses[i] == 'cohort_cls':
                continue
            elif 'epithelium' in responses[i]: # TODO: BINARY
                continue # TODO: BINARY
            labels = torch.stack([wsi_labels[j][i] for j in range(len(wsi_labels))])
            output = output_dict[responses[i]][0]
            
            #print('Labels size:', labels.shape)
            #print('Output size:', output.shape)

            if responses[i] in ['CMS_matching', 'CMS']: # TODO: BINARY
                loss += loss_weights[i] * multiclass_criterion(output.squeeze(),  # TODO: BINARY
                                                               labels.squeeze().type(torch.LongTensor).cuda())
            elif loss_name == 'slidegraph':
                labels_ = labels[:, None]
                labels_ = labels_ - labels_.T
                output_ = output - output.T
                diff = output_[labels_ > 0]
                resp_loss = torch.mean(F.relu(1.0 - diff))
                loss += loss_weights[i] * resp_loss
            elif loss_name == 'bce':
                # node_output_ = node_output.squeeze().cuda()
                # labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
                loss += loss_weights[i] * criterion(output.squeeze(), labels.squeeze().type(torch.FloatTensor).cuda())
        
        if 'cohort_cls' in responses:
            cohort_idx = responses.index('cohort_cls')
        #if responses[2] == 'cohort_cls':
            labels = torch.stack([wsi_labels[j][cohort_idx] for j in range(len(wsi_labels))])
            output = output_dict[responses[cohort_idx]][0]

            # Negative loss for cohort - don't want to be able to predict
            cohort_loss = loss_weights[cohort_idx] * multiclass_criterion(output.squeeze(),
                                                             labels.squeeze().type(torch.FloatTensor).cuda())
            #loss -= cohort_loss

            print('Cohort training loss:', cohort_loss)

            epi_label_idx = cohort_idx + 1

        if any('epithelium' in resp for resp in responses):
            epi_label_idx = [idx for idx, s in enumerate(responses) if 'epi' in s][0]
            # For epithelial response
            node_output = output_dict[responses[-1]][1]
            labels = torch.cat([wsi_labels[j][epi_label_idx:] for j in range(len(wsi_labels))])  # cat flattens lists
    
            if loss_name == 'slidegraph':
                labels = labels.reshape(len(labels), 1)
                # wsi_output = flat_logit.reshape(len(flat_logit),1)
    
                n_splits = 10
                node_output_n = np.array_split(node_output, n_splits)
                labels_n = np.array_split(labels, n_splits)
    
                diff = torch.Tensor([]).cuda()
                for i in range(n_splits):
                    node_output_i = node_output_n[i]
                    labels_i = labels_n[i]
    
                    node_output_ = node_output_i - node_output_i.T
                    labels_ = labels_i - labels_i.T
                    del node_output_i, labels_i
    
                    diff = torch.cat((diff, (node_output_[labels_ > 0])))
                    del node_output_, labels_
    
                loss += loss_weights[-1] * torch.mean(F.relu(1.0 - diff))

            elif loss_name == 'bce':
                node_output_ = node_output.squeeze().cuda()
                labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
                loss += loss_weights[-1] * criterion(node_output_, labels_)
            else:
                raise Exception('loss not defined')

        #if responses[2] == 'cohort_cls':
        if 'cohort_cls' in responses:
            loss = loss / cohort_loss
            #loss += cohort_loss

        if temper is not None:
            loss = loss * temper

        # Backprop and update
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = [wsi_label.cpu().numpy() for wsi_label in wsi_labels]
        return [loss, output_dict, wsi_labels]

    # Run one inference step
    @staticmethod
    def infer_batch(model, batch_data):
        wsi_graphs = batch_data["graph"].to("cuda")

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            output_dict = model(wsi_graphs)  # contains wsi and node predictions

        # Output should be a single tensor or scalar
        if "label" in batch_data:
            wsi_labels = batch_data["label"]
            wsi_labels = wsi_labels.cpu().numpy()
            return output_dict, wsi_labels
        return output_dict, _
        # return [output_dict]


# linear, pooling, dropout -> GNN convs -> linear, pooling, dropout + residuals -> MLP v2 with 1 layer
# -> linear, pooling, dropout
class SlideGraphArchMLPv2(nn.Module):
    def __init__(
            self,
            responses,
            dim_features,
            dim_target,
            layers=[6, 6],
            pooling="max",
            dropout=0.0,
            conv="GINConv",
            gembed=False,
            scale=False,
            temper=None,
            use_mlp=True,
            mlp_dropout=0.1,
            label_dim=[1, 1, 1],
            **kwargs
    ):
        super().__init__()
        self.responses = responses
        self.dropout = dropout
        self.embeddings_dim = layers
        self.num_layers = len(self.embeddings_dim)
        self.convs = []
        self.linears = []
        self.branch_linears = []
        #self.mlp_heads = []
        self.mlpv2 = []
        self.pooling = {
            "max": global_max_pool,
            "mean": global_mean_pool,
            "add": global_add_pool,
        }[pooling]
        # If True then learn a graph embedding for final classification
        # (classify pooled node features), otherwise pool node decision scores.
        self.gembed = gembed
        self.temper = temper
        self.use_mlp = use_mlp
        self.label_dim = label_dim
        assert len(self.label_dim) == len(self.responses), \
            "Binary list and responses must be equal length"

        conv_dict = {"GINConv": [GINConv, 1], "EdgeConv": [EdgeConv, 2], "GATConv": [GATConv, 1]}  # changed from 1 to 2
        if conv not in conv_dict:
            raise ValueError(f'Not support `conv="{conv}".')

        def create_linear(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims), BatchNorm1d(out_dims), ReLU()
            )

        input_emb_dim = dim_features
        out_emb_dim = self.embeddings_dim[0]
        self.first_h = create_linear(input_emb_dim, out_emb_dim)
        self.linears.append(Linear(out_emb_dim, dim_target))

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embeddings_dim[1:]:
            ConvClass, alpha = conv_dict[conv]
            if conv == 'GATConv':
                self.convs.append(ConvClass(in_channels=alpha * input_emb_dim,
                                            out_channels=out_emb_dim, **kwargs))
            else:
                subnet = create_linear(alpha * input_emb_dim, out_emb_dim)

                self.convs.append(ConvClass(subnet, **kwargs))
            self.linears.append(Linear(out_emb_dim, dim_target))
            input_emb_dim = out_emb_dim

        # TODO: add MLP conv for final layer, x3 for each response

        for i in range(len(responses)):
            self.branch_linears.append(Linear(self.embeddings_dim[-1], self.label_dim[i]))
            #self.mlp_heads.append(nn.Sequential(
            #    nn.BatchNorm1d(dim_target),
            #    nn.Linear(dim_target, 1),
            #    nn.Sigmoid()))

            #self.mlpv2.append(ops.MLP(in_channels=self.embeddings_dim[-1],
            #                          hidden_channels = [int(self.embeddings_dim[-1]/2), dim_target],
            #                          norm_layer=nn.LayerNorm,
            #                          activation_layer=nn.modules.activation.LeakyReLU,
            #                          bias=True,
            #                          dropout=0.1))

            #self.mlpv2.append(MLPAggregation(in_channels=self.embeddings_dim[-2],
            #                                out_channels=int(self.embeddings_dim[-1]),
            #                                max_num_elements=1,
            #                                 # Could add bigger and smaller layers or [32, 24, 16]
            #                                hidden_channels=None, # [int(self.embeddings_dim[-1])],
            #                                norm="batch_norm",
            #                                act="relu",
            #                                bias=True,
            #                                dropout=0.1))

            #print(f'Adding MLP with dropout {mlp_dropout}')
            
            # This is just a linear layer 16 -> 8
            self.mlpv2.append(MLP(in_channels=self.embeddings_dim[-2],
                                  # hidden_channels=int((in_channels * max_num_elements) / 2),
                                  out_channels=label_dim[i],
                                  num_layers=1,
                                  hidden_channels=None,  # [int(self.embeddings_dim[-1])],
                                  norm="batch_norm",
                                  act="relu",
                                  bias=True,
                                  dropout=mlp_dropout))

            # not used
            #self.mlpv3.append(MLP(channel_list = [self.embeddings_dim[-2], self.embeddings_dim[-1], 1],
            #                      norm="batch_norm",
            #                      act="relu",
            #                      bias=True,
            #                      dropout=0.1))

            # self.branch_linears.append(Linear(self.embeddings_dim[-1], int(self.embeddings_dim[-1]/2)))

            # self.mlp_heads.append(torchvision.ops.MLP(in_channels=int(self.embeddings_dim[-1]),
            #                             hidden_channels=[int(self.embeddings_dim[-1]/2), dim_target],
            #                             norm_layer = nn.BatchNorm1d,
            #                             activation_layer = torch.nn.LeakyReLU,
            #                             inplace = False,
            #                             bias = True,
            #                             dropout = 0.1))
            # nn.Sigmoid()))

        self.sig = nn.Sigmoid()
        self.scale = scale

        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)
        self.branch_linears = torch.nn.ModuleList(self.branch_linears)
        #self.mlp_heads = torch.nn.ModuleList(self.mlp_heads)  # this puts it onto cuda
        self.mlpv2 = torch.nn.ModuleList(self.mlpv2)  # this puts it onto cuda
        #self.dim_target = dim_target

        # Auxilary holder for external model, these are saved separately from torch.save
        # as they can be sklearn model etc.
        self.aux_model = {}

    def save(self, path, aux_path=None):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        if aux_path is not None:
            joblib.dump(self.aux_model, aux_path)

    def load(self, path, aux_path=None):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        if aux_path is not None:
            self.aux_model = joblib.load(aux_path)

    def forward(self, data):

        edge_index = data.edge_index
        if edge_index.dtype == torch.float64:
            edge_index = data.edge_index.long()
        feature, batch = data.x, data.batch

        wsi_prediction = 0
        pooling = self.pooling
        node_prediction = 0
        out_dict = OrderedDict()

        feature = self.first_h(feature)
        for layer in range(self.num_layers):
            if layer == 0:
                # First layer, linear, pooling and dropout only
                node_prediction_sub = self.linears[layer](feature)
                node_prediction += node_prediction_sub
                node_pooled = pooling(node_prediction_sub, batch)
                wsi_prediction_sub = F.dropout(
                    node_pooled, p=self.dropout, training=self.training
                )
                wsi_prediction += wsi_prediction_sub
            elif layer == self.num_layers - 1:
                # Final layer, branch for each response in output
                #branches_feature = self.convs[layer - 1](feature, edge_index)
                #print('Features in final layer:', branches_feature.shape)

                #Features in middle layers: torch.Size([406164, 32])
                #Node_prediction_sub in middle layers: torch.Size([406164, 1])
                #Features in final layer: torch.Size([406164, 16])
                #Node prediction in final layer: torch.Size([406164, 1])


                for i in range(len(self.responses)):
                    _wsi_prediction = wsi_prediction.clone()
                    _node_prediction = node_prediction.clone()
                    #print('Node prediction in final layer:', _node_prediction.shape) #[n, 1]

                    if self.use_mlp:
                        ### Removed branch_linears below from v7.1 onwards, just using MLP (not both).
                        #branches_feature = self.mlpv2[i](feature)
                        node_prediction_sub = self.mlpv2[i](feature)
                        branches_feature = node_prediction_sub
                    else:
                        branches_feature = self.convs[layer - 1](feature, edge_index)
                        node_prediction_sub = self.branch_linears[i](branches_feature)

                    #print('Features in final layer after MLPAggregation:', branches_feature.shape) # [1,1]

                    if not self.gembed:
                        #node_prediction_sub = self.branch_linears[i](branches_feature)
                        _node_prediction += node_prediction_sub
                        node_pooled = pooling(node_prediction_sub, batch)
                        wsi_prediction_sub = F.dropout(
                            node_pooled, p=self.dropout, training=self.training
                        )
                    else:
                        node_pooled = pooling(branches_feature, batch)
                        #if self.use_mlp:
                        #    node_prediction_sub = self.mlpv2[i](node_pooled)
                        #else:
                        node_prediction_sub = self.branch_linears[i](node_pooled)
                        wsi_prediction_sub = F.dropout(
                            node_prediction_sub, p=self.dropout, training=self.training
                        )
                    
                    #_wsi_prediction += wsi_prediction_sub 
                    _wsi_prediction = wsi_prediction_sub # try without adding previous predictions

                    if not self.scale:
                        if self.temper is not None:
                            _wsi_prediction = _wsi_prediction / self.temper
                            _node_prediction = _node_prediction / self.temper
                        if self.label_dim[i] == 1:
                            _wsi_prediction = self.sig(_wsi_prediction)
                            _node_prediction = self.sig(_node_prediction)
                        # if want probabilities, use softmax. For CE loss, use raw logits.
                    out_dict[self.responses[i]] = [_wsi_prediction, _node_prediction]

                # Returning features as well
                out_dict['features'] = feature
            else:
                # All other layers, with GinConv then linear, pooling and dropout
                feature = self.convs[layer - 1](feature, edge_index)
                #print('Features in middle layers:', feature.shape)
                if not self.gembed:
                    node_prediction_sub = self.linears[layer](feature)
                    #print('Node prediction sub in middle layers:', node_prediction_sub.shape)
                    node_prediction += node_prediction_sub
                    node_pooled = pooling(node_prediction_sub, batch)
                    wsi_prediction_sub = F.dropout(
                        node_pooled, p=self.dropout, training=self.training
                    )
                else:
                    node_pooled = pooling(feature, batch)
                    node_prediction_sub = self.linears[layer](node_pooled)
                    wsi_prediction_sub = F.dropout(
                        node_prediction_sub, p=self.dropout, training=self.training
                    )
                wsi_prediction += wsi_prediction_sub
        return out_dict

    # output dict with {'response_cr_nocr': [wsi_pred, node_pred], 'CMS4': [wsi_pred, node_pred], etc.}

    # Run one single step
    @staticmethod
    def train_batch(model, batch_data, responses, loss_name, loss_weights, optimizer: torch.optim.Optimizer,
                    criterion=None, temper=None):
        wsi_graphs = batch_data["graph"].to("cuda")
        wsi_labels = batch_data["label"].to("cuda")  # both labels for both responses like [0, 1]

        # remove padding from labels
        orig_lengths = batch_data["length"]  # .to("cuda")
        wsi_labels = [wsi_labels[i, :orig_lengths[i]] for i in range(len(orig_lengths))]

        # WSI GRAPHS: DataBatch(x=[589, 384], edge_index=[2, 3366], coords=[589, 2], batch=[589], ptr=[4])
        # WSI LABELS: tensor([1., 1., 1.], device='cuda:0', dtype=torch.float64)

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Not an RNN so does not accumulate
        model.train()
        optimizer.zero_grad()

        output_dict = model(wsi_graphs)

        multiclass_criterion = nn.CrossEntropyLoss().cuda()

        loss = 0
        for i in range(len(responses)): # works for less than 2 responses also
            if responses[i] == 'cohort_cls':
                continue
            elif 'epithelium' in responses[i]:
                continue
            labels = torch.stack([wsi_labels[j][i] for j in range(len(wsi_labels))])
            output = output_dict[responses[i]][0]

            if responses[i] in ['CMS_matching', 'CMS']:
                loss += loss_weights[i] * multiclass_criterion(output.squeeze(),
                                                               labels.squeeze().type(torch.LongTensor).cuda())
            elif loss_name == 'slidegraph':
                labels_ = labels[:, None]
                labels_ = labels_ - labels_.T
                output_ = output - output.T
                diff = output_[labels_ > 0]
                resp_loss = torch.mean(F.relu(1.0 - diff))
                loss += loss_weights[i] * resp_loss
            elif loss_name == 'bce':
                # node_output_ = node_output.squeeze().cuda()
                # labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
                loss += loss_weights[i] * criterion(output.squeeze(), labels.squeeze().type(torch.FloatTensor).cuda())
        
        if 'cohort_cls' in responses:
            cohort_idx = responses.index('cohort_cls')
        #if responses[2] == 'cohort_cls':
            labels = torch.stack([wsi_labels[j][cohort_idx] for j in range(len(wsi_labels))])
            output = output_dict[responses[cohort_idx]][0]

            # Negative loss for cohort - don't want to be able to predict
            cohort_loss = loss_weights[cohort_idx] * multiclass_criterion(output.squeeze(),
                                                             labels.squeeze().type(torch.FloatTensor).cuda())
            #loss -= cohort_loss

            print('Cohort training loss:', cohort_loss)

            epi_label_idx = cohort_idx + 1

        if any('epithelium' in resp for resp in responses):
            epi_label_idx = [idx for idx, s in enumerate(responses) if 'epi' in s][0]
            # For epithelial response
            node_output = output_dict[responses[-1]][1]
            labels = torch.cat([wsi_labels[j][epi_label_idx:] for j in range(len(wsi_labels))])  # cat flattens lists
    
            if loss_name == 'slidegraph':
                labels = labels.reshape(len(labels), 1)
                # wsi_output = flat_logit.reshape(len(flat_logit),1)
    
                n_splits = 10
                node_output_n = np.array_split(node_output, n_splits)
                labels_n = np.array_split(labels, n_splits)
    
                diff = torch.Tensor([]).cuda()
                for i in range(n_splits):
                    node_output_i = node_output_n[i]
                    labels_i = labels_n[i]
    
                    node_output_ = node_output_i - node_output_i.T
                    labels_ = labels_i - labels_i.T
                    del node_output_i, labels_i
    
                    diff = torch.cat((diff, (node_output_[labels_ > 0])))
                    del node_output_, labels_
    
                loss += loss_weights[-1] * torch.mean(F.relu(1.0 - diff))

            elif loss_name == 'bce':
                node_output_ = node_output.squeeze().cuda()
                labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
                loss += loss_weights[-1] * criterion(node_output_, labels_)
            else:
                raise Exception('loss not defined')

        #if responses[2] == 'cohort_cls':
        if 'cohort_cls' in responses:
            loss = loss / cohort_loss
            #loss += cohort_loss

        if temper is not None:
            loss = loss * temper

        # Backprop and update
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = [wsi_label.cpu().numpy() for wsi_label in wsi_labels]
        return [loss, output_dict, wsi_labels]

    # Run one inference step
    @staticmethod
    def infer_batch(model, batch_data):
        wsi_graphs = batch_data["graph"].to("cuda")

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            output_dict = model(wsi_graphs)  # contains wsi and node predictions

        # Output should be a single tensor or scalar
        if "label" in batch_data:
            wsi_labels = batch_data["label"]
            wsi_labels = wsi_labels.cpu().numpy()
            return output_dict, wsi_labels
        return output_dict, _
        # return [output_dict]


class SlideGraphArch(nn.Module):
    def __init__(
            self,
            responses,
            dim_features,
            dim_target,
            layers=[6, 6],
            pooling="max",
            dropout=0.0,
            conv="GINConv",
            gembed=False,
            scale=False,
            temper=None,
            use_mlp=True,
            **kwargs
    ):
        super().__init__()
        self.responses = responses
        self.dropout = dropout
        self.embeddings_dim = layers
        self.num_layers = len(self.embeddings_dim)
        self.convs = []
        self.linears = []
        self.branch_linears = []
        self.mlp_heads = []
        self.pooling = {
            "max": global_max_pool,
            "mean": global_mean_pool,
            "add": global_add_pool,
        }[pooling]
        # If True then learn a graph embedding for final classification
        # (classify pooled node features), otherwise pool node decision scores.
        self.gembed = gembed
        self.temper = temper
        self.use_mlp = use_mlp

        conv_dict = {"GINConv": [GINConv, 1], "EdgeConv": [EdgeConv, 2]}  # changed from 1 to 2
        if conv not in conv_dict:
            raise ValueError(f'Not support `conv="{conv}".')

        def create_linear(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims), BatchNorm1d(out_dims), ReLU()
            )

        input_emb_dim = dim_features
        out_emb_dim = self.embeddings_dim[0]
        self.first_h = create_linear(input_emb_dim, out_emb_dim)
        self.linears.append(Linear(out_emb_dim, dim_target))

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embeddings_dim[1:]:
            ConvClass, alpha = conv_dict[conv]
            if conv == 'GATConv':
                self.convs.append(ConvClass(in_channels=alpha * input_emb_dim,
                                            out_channels=out_emb_dim, **kwargs))
            else:
                subnet = create_linear(alpha * input_emb_dim, out_emb_dim)

                self.convs.append(ConvClass(subnet, **kwargs))
            self.linears.append(Linear(out_emb_dim, dim_target))
            input_emb_dim = out_emb_dim

        for i in range(len(responses)):
            self.branch_linears.append(Linear(self.embeddings_dim[-1], dim_target))
            self.mlp_heads.append(nn.Sequential(
                nn.BatchNorm1d(dim_target),
                nn.Linear(dim_target, 1),
                nn.Sigmoid()))


            # self.branch_linears.append(Linear(self.embeddings_dim[-1], int(self.embeddings_dim[-1]/2)))

            # self.mlp_heads.append(torchvision.ops.MLP(in_channels=int(self.embeddings_dim[-1]),
            #                             hidden_channels=[int(self.embeddings_dim[-1]/2), dim_target],
            #                             norm_layer = nn.BatchNorm1d,
            #                             activation_layer = torch.nn.LeakyReLU,
            #                             inplace = False,
            #                             bias = True,
            #                             dropout = 0.1))
            # nn.Sigmoid()))

        self.sig = nn.Sigmoid()
        self.scale = scale

        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)
        self.branch_linears = torch.nn.ModuleList(self.branch_linears)
        self.mlp_heads = torch.nn.ModuleList(self.mlp_heads)  # this puts it onto cuda

        # Auxilary holder for external model, these are saved separately from torch.save
        # as they can be sklearn model etc.
        self.aux_model = {}

    def save(self, path, aux_path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        joblib.dump(self.aux_model, aux_path)

    def load(self, path, aux_path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.aux_model = joblib.load(aux_path)

    def forward(self, data):

        edge_index = data.edge_index
        if edge_index.dtype == torch.float64:
            edge_index = data.edge_index.long()
        feature, batch = data.x, data.batch

        wsi_prediction = 0
        pooling = self.pooling
        node_prediction = 0
        out_dict = OrderedDict()

        feature = self.first_h(feature)
        for layer in range(self.num_layers):
            if layer == 0:
                # First layer, linear, pooling and dropout only
                node_prediction_sub = self.linears[layer](feature)
                node_prediction += node_prediction_sub
                node_pooled = pooling(node_prediction_sub, batch)
                wsi_prediction_sub = F.dropout(
                    node_pooled, p=self.dropout, training=self.training
                )
                wsi_prediction += wsi_prediction_sub
            elif layer == self.num_layers - 1:
                # Final layer, branch for each response in output
                branches_feature = self.convs[layer - 1](feature, edge_index)
                for i in range(len(self.responses)):
                    _wsi_prediction = wsi_prediction.clone()
                    _node_prediction = node_prediction.clone()

                    if not self.gembed:
                        node_prediction_sub = self.branch_linears[i](branches_feature)
                        _node_prediction += node_prediction_sub
                        node_pooled = pooling(node_prediction_sub, batch)
                        wsi_prediction_sub = F.dropout(
                            node_pooled, p=self.dropout, training=self.training
                        )
                    else:
                        node_pooled = pooling(branches_feature, batch)
                        node_prediction_sub = self.branch_linears[i](node_pooled)
                        wsi_prediction_sub = F.dropout(
                            node_prediction_sub, p=self.dropout, training=self.training
                        )
                    _wsi_prediction += wsi_prediction_sub

                    if not self.scale:
                        if self.temper is not None:
                            _wsi_prediction = _wsi_prediction / self.temper
                            _node_prediction = _node_prediction / self.temper
                        if self.use_mlp:
                            _wsi_prediction = self.mlp_heads[i](_wsi_prediction)
                            _node_prediction = self.mlp_heads[i](_node_prediction)
                        else:
                            _wsi_prediction = self.sig(_wsi_prediction)
                            _node_prediction = self.sig(_node_prediction)
                    out_dict[self.responses[i]] = [_wsi_prediction, _node_prediction]

                ## Returning features as well
                #out_dict['features'] = feature
            else:
                # All other layers, with GinConv then linear, pooling and dropout
                feature = self.convs[layer - 1](feature, edge_index)
                if not self.gembed:
                    node_prediction_sub = self.linears[layer](feature)
                    node_prediction += node_prediction_sub
                    node_pooled = pooling(node_prediction_sub, batch)
                    wsi_prediction_sub = F.dropout(
                        node_pooled, p=self.dropout, training=self.training
                    )
                else:
                    node_pooled = pooling(feature, batch)
                    node_prediction_sub = self.linears[layer](node_pooled)
                    wsi_prediction_sub = F.dropout(
                        node_prediction_sub, p=self.dropout, training=self.training
                    )
                wsi_prediction += wsi_prediction_sub
        return out_dict

    # output dict with {'response_cr_nocr': [wsi_pred, node_pred], 'CMS4': [wsi_pred, node_pred], etc.}

    # Run one single step
    @staticmethod
    def train_batch(model, batch_data, responses, loss_name, loss_weights, optimizer: torch.optim.Optimizer,
                    criterion=None, temper=None):
        wsi_graphs = batch_data["graph"].to("cuda")
        wsi_labels = batch_data["label"].to("cuda")  # both labels for both responses like [0, 1]

        # remove padding from labels
        orig_lengths = batch_data["length"]  # .to("cuda")
        wsi_labels = [wsi_labels[i, :orig_lengths[i]] for i in range(len(orig_lengths))]

        # WSI GRAPHS: DataBatch(x=[589, 384], edge_index=[2, 3366], coords=[589, 2], batch=[589], ptr=[4])
        # WSI LABELS: tensor([1., 1., 1.], device='cuda:0', dtype=torch.float64)

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Not an RNN so does not accumulate
        model.train()
        optimizer.zero_grad()

        output_dict = model(wsi_graphs)

        loss = 0
        for i in range(len(responses[:2])):
            labels = torch.stack([wsi_labels[j][i] for j in range(len(wsi_labels))])
            output = output_dict[responses[i]][0]

            if loss_name == 'slidegraph':
                labels_ = labels[:, None]
                labels_ = labels_ - labels_.T
                output_ = output - output.T
                diff = output_[labels_ > 0]
                resp_loss = torch.mean(F.relu(1.0 - diff))
                loss += loss_weights[i] * resp_loss
            elif loss_name == 'bce':
                # node_output_ = node_output.squeeze().cuda()
                # labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
                loss += loss_weights[i] * criterion(output.squeeze(), labels.squeeze().type(torch.FloatTensor).cuda())

        # For epithelial response
        node_output = output_dict[responses[-1]][1]
        labels = torch.cat([wsi_labels[j][2:] for j in range(len(wsi_labels))])  # cat flattens lists

        if loss_name == 'slidegraph':
            labels = labels.reshape(len(labels), 1)
            # wsi_output = flat_logit.reshape(len(flat_logit),1)

            n_splits = 10
            node_output_n = np.array_split(node_output, n_splits)
            labels_n = np.array_split(labels, n_splits)

            diff = torch.Tensor([]).cuda()
            for i in range(n_splits):
                node_output_i = node_output_n[i]
                labels_i = labels_n[i]

                node_output_ = node_output_i - node_output_i.T
                labels_ = labels_i - labels_i.T
                del node_output_i, labels_i

                diff = torch.cat((diff, (node_output_[labels_ > 0])))
                del node_output_, labels_

            loss += loss_weights[-1] * torch.mean(F.relu(1.0 - diff))

        elif loss_name == 'bce':
            node_output_ = node_output.squeeze().cuda()
            labels_ = labels.squeeze().type(torch.FloatTensor).cuda()
            loss += loss_weights[-1] * criterion(node_output_, labels_)
        else:
            raise Exception('loss not defined')

        if temper is not None:
            loss = loss * temper

        # Backprop and update
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = [wsi_label.cpu().numpy() for wsi_label in wsi_labels]
        return [loss, output_dict, wsi_labels]

    # Run one inference step
    @staticmethod
    def infer_batch(model, batch_data):
        wsi_graphs = batch_data["graph"].to("cuda")

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            output_dict = model(wsi_graphs)  # contains wsi and node predictions

        # Output should be a single tensor or scalar
        if "label" in batch_data:
            wsi_labels = batch_data["label"]
            wsi_labels = wsi_labels.cpu().numpy()
            return output_dict, wsi_labels
        return output_dict, _
        # return [output_dict]


def slidegraph_loss_function(labels, output):  # pass wsi_labels[:,0] and wsi_labels[:,1] separately
    labels_ = labels[:, None]
    labels_ = labels_ - labels_.T
    output_ = output - output.T
    diff = output_[labels_ > 0]
    loss = torch.mean(F.relu(1.0 - torch.Tensor(diff)))
    return loss


def select_checkpoints(
        stat_file_path: str,
        top_k: int = 2,
        metrics: [str] = ["infer-valid-auprc"], # changed to array
        mean: bool = True,
        epoch_range: Tuple[int] = [0, 1000],
):
    """Select checkpoints basing on training statistics.

    Args:
        stat_file_path (str): Path pointing to the .json
            which contains the statistics.
        top_k (int): Number of top checkpoints to be selected.
        metric (str): The metric name saved within .json to perform
            selection.
        epoch_range (list): The range of epochs for checking, denoted
            as [start, end] . Epoch x that is `start <= x <= end` is
            kept for further selection.
    Returns:
        paths (list): List of paths or info tuple where each point
            to the correspond check point saving location.
        stats (list): List of corresponding statistics.

    """
    stats_dict = load_json(stat_file_path)

    # IF DOESN'T FINISH TRAINING
    # stats_dict.pop('9')

    # k is the epoch counter in this case
    stats_dict = {
        k: v
        for k, v in stats_dict.items()
        if int(k) >= epoch_range[0] and int(k) <= epoch_range[1]
    }
    if mean:
        stats = [[int(k), 
                  np.mean([v[met_k] for met_k in v.keys() if any([metric in met_k for metric in metrics])]),
                  v] for k,v in stats_dict.items()]
    else:
        stats = [[int(k), [v[met_k] for met_k in v.keys() if any([metric in met_k for metric in metrics])], 
                  v] for k, v in stats_dict.items()]
        
        #stats = [[int(k), v[metric], v] for k, v in stats_dict.items()]
    # sort epoch ranking from largest to smallest
    stats = sorted(stats, key=lambda v: v[1], reverse=True)
    chkpt_stats_list = stats[:top_k]  # select top_k

    model_dir = pathlib.Path(stat_file_path).parent
    epochs = [v[0] for v in chkpt_stats_list]
    paths = [
        (
            f"{model_dir}/epoch={epoch:03d}.weights.pth",
            f"{model_dir}/epoch={epoch:03d}.aux.dat",
        )
        for epoch in epochs
    ]
    chkpt_stats_list = [[v[0], v[2]] for v in chkpt_stats_list]
    print(paths)
    return paths, chkpt_stats_list, epochs
