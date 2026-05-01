import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """
    KoLeo regularization loss from DINOv2.
    Encourages uniform distribution of features on the unit hypersphere
    by maximizing the minimum pairwise distance within the batch.

    Reference: Sablayrolles et al., "Spreading vectors for similarity search" (2018)
               Oquab et al., DINOv2 (2023)
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def _pairwise_NNs_inner(self, x):
        """
        Find nearest neighbor index for each sample using inner product.
        For L2-normalized vectors, inner product == cosine similarity.
        Args:
            x: [N, D] L2-normalized features
        Returns:
            nn_inds: [N] nearest neighbor indices (excluding self)
        """
        dots = torch.mm(x, x.t())  # [N, N]
        n = x.shape[0]
        # Mask diagonal with -1 so self is never the nearest neighbor
        dots.view(-1)[:: n + 1].fill_(-1)
        _, nn_inds = torch.max(dots, dim=1)
        return nn_inds

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output: [N, D] backbone CLS token features (before DINO head)
        Returns:
            scalar KoLeo loss
        """
        with torch.cuda.amp.autocast(enabled=False):
            # L2 normalize
            x = F.normalize(student_output.float(), p=2, dim=-1, eps=eps)
            # Find nearest neighbors
            nn_inds = self._pairwise_NNs_inner(x)
            # Euclidean distances to nearest neighbors
            distances = self.pdist(x, x[nn_inds])
            loss = -torch.log(distances + eps).mean()
        return loss
