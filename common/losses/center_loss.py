import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, mean=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = mean.cuda()
        else:
            self.centers = mean
        self.x = nn.Parameter(torch.zeros(256, 128).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        self.x = x
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # print(self.centers[0][0])
        # print(self.x[0][0])
        return loss

    def classify(self, z_unmasked, target):
        centroid_real = self.centers[0]
        centroid_fake = self.centers[1]
        print(torch.mean(centroid_real))
        real_vec = torch.mean(torch.abs(z_unmasked - centroid_real), dim=1)
        fake_vec = torch.mean(torch.abs(z_unmasked - centroid_fake), dim=1)
        # print(torch.max(torch.cat([real_vec.unsqueeze(1), fake_vec.unsqueeze(1)], dim=1), dim=1).shape)
        _, y_pred = torch.max(torch.cat([real_vec.unsqueeze(1), fake_vec.unsqueeze(1)], dim=1), dim=1)
        target_center = target.clone().detach()
        target_center[target_center == 2] = 1
        target_center[target_center == 3] = 1
        return (y_pred == target_center).sum()
