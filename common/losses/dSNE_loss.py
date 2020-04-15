import torch
import torch.nn as nn
import torch.nn.functional as F

class dSNELoss(nn.Module):
    """
    dSNE Loss
    """
    def __init__(self, margin=1):
        super(dSNELoss, self).__init__()
        # self._bs_src = bs_src
        # self._bs_tgt = bs_tgt
        # self._embed_size = embed_size
        self._margin = margin
        # self._fn = fn

    def forward(self, fts, ys, ftt, yt):
        """
        Semantic Alignment Loss
        :param fts: features for the source domain [M, K]
        :param ys: label for the source domain [M]
        :param ftt: features for the target domain [N, K]
        :param yt: label for the target domain [N]
        :return:
        """
        # if self._fn:
        #     # Normalize embedding space
        #     fts_norm = F.normalize(input=fts, p=2, dim=1)
        #     ftt_norm = F.normalize(input=ftt, p=2, dim=1)

        fts_rpt = fts.unsqueeze(dim=0)
        ftt_rpt = ftt.unsqueeze(dim=1)

        dists = torch.sum((ftt_rpt - fts_rpt)**2, dim=2)

        yt_rpt = yt.unsqueeze(dim=1)
        ys_rpt = ys.unsqueeze(dim=0)

        y_same = torch.eq(yt_rpt, ys_rpt).float()
        y_diff = torch.ne(yt_rpt, ys_rpt).float()

        intra_cls_dists = dists * y_same
        inter_cls_dists = dists * y_diff

        max_dists, _ = torch.max(dists, dim=1)
        revised_inter_cls_dists = torch.where(y_same.byte(), max_dists, inter_cls_dists)

        max_intra_cls_dist, _ = torch.max(intra_cls_dists, dim=1)
        min_inter_cls_dist, _ = torch.min(revised_inter_cls_dists, dim=1)

        loss = torch.relu(max_intra_cls_dist - min_inter_cls_dist + self._margin)

        return loss