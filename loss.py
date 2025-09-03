import torch

class ClusterTripletLoss(torch.nn.Module):
    def __init__(self):
        super(ClusterTripletLoss, self).__init__()

    def forward(self, input_features, centroids):
        # input_features is shape (n_samples, n_features)
        # centroids is shape (k_clusters, n_features)

        assert (input_features.shape[1] == centroids.shape[1]), "Dimensions Mismatch"

        positives = torch.tensor([], device='cuda', requires_grad=True)
        negatives = torch.tensor([], device='cuda', requires_grad=True)
        for feature_sample in input_features:
            closest = centroids[torch.nn.functional.mse_loss(feature_sample,
                                                             centroids,
                                                             reduce=False).min(dim=0)[1].mode()[0]]
            print(closest.grad_fn)
            furthest = centroids[torch.nn.functional.mse_loss(feature_sample,
                                                              centroids,
                                                              reduce=False).max(dim=0)[1].mode()[0]]
            print(furthest.grad_fn)

            # anchor = torch.cat[]
            # anchor = feature_sample
            positives = torch.cat((positives, closest.unsqueeze(0)))
            negatives = torch.cat((negatives, furthest.unsqueeze(0)))
        print(positives.grad_fn)
        print(negatives.grad_fn)
        print(input_features.grad_fn)
        loss = torch.nn.functional.triplet_margin_loss(input_features, positives, negatives, margin=1, swap=True)
        # print(loss.grad_fn)
        return loss

