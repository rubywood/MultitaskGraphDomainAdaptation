import torch


class ClusterTripletLoss(torch.nn.Module):
    def __init__(self, centroids):
        super(ClusterTripletLoss, self).__init__()
        self.centroids = centroids

    def forward(self, input_features):
        # input_features is shape (n_samples, n_features)
        # centroids is shape (k_clusters, n_features)

        assert (input_features.shape[1] == self.centroids.shape[1]) ,"Dimensions Mismatch"

        distances = torch.nn.functional.mse_loss(input_features.unsqueeze(1).repeat(1,
                                                                                    self.centroids.shape[0], 1),
                                                 self.centroids, reduce=False)
        positives = self.centroids[distances.min(dim=1)[1].mode()[0]]
        negatives = self.centroids[distances.max(dim=1)[1].mode()[0]]

        loss = torch.nn.functional.triplet_margin_loss(input_features, positives, negatives, margin=1, swap=True)
        return loss


class CentreTripletLoss(torch.nn.Module):
    # Inspiration from Triplet-center loss for multi-view 3d object retrieval, 2018, Xinwei He
    # Use second closest cluster centre as negative example in triplet loss

    def __init__(self, centroids):
        super(CentreTripletLoss, self).__init__()
        self.centroids = centroids

    def forward(self, input_features):
        # input_features is shape (n_samples, n_features)
        # centroids is shape (k_clusters, n_features)

        assert (input_features.shape[1] == self.centroids.shape[1]), "Dimensions Mismatch"

        distances = torch.nn.functional.mse_loss(input_features.unsqueeze(1).repeat(1,
                                                                                    self.centroids.shape[0], 1),
                                                 self.centroids, reduce=False)  # torch.Size([10100, 6, 16])

        closest = torch.topk(distances, k=2, dim=1, largest=False, sorted=True)[1]  # torch.Size([10100, 2, 16])

        positives = self.centroids[closest[:, 0, :].mode()[0]]  # torch.Size([10100, 16])
        negatives = self.centroids[closest[:, 1, :].mode()[0]]  ##torch.Size([10100, 16])

        loss = torch.nn.functional.triplet_margin_loss(input_features, positives, negatives, margin=1, swap=True)
        return loss