"""
Custom Loss implementations
"""
import torch
import torch.nn.functional as F
from common.utils.common_utils import euclidean_dist

m = torch.nn.Softmax(dim=1)
var_inv = 1


def ActivationLoss(activation_vector, y_onehot, weight_vector):
    act_loss = torch.mean(weight_vector.unsqueeze(1).cuda() * torch.abs(activation_vector - y_onehot))
    return act_loss


def ActivationLossFineTune(activation_vector, y_onehot):
    l_vec = torch.abs(m(activation_vector) - y_onehot)
    l_real = l_vec[:, 0]
    l_fake_sum = torch.sum(l_vec[:, 1:], dim=1)
    l_fake_max, _ = torch.max(l_vec[:, 1:], dim=1)
    l_fake_min, _ = torch.min(l_vec[:, 1:], dim=1)
    l_fake = y_onehot[:, 0] * l_fake_sum + y_onehot[:, 1] * l_fake_min
    act_loss = torch.sum(torch.stack((l_real, l_fake), dim=1))
    return act_loss


def kl_with_gaussian_custom_std(mu, logvar, mean):
    kl_div = -0.5 * torch.sum(1 + logvar - var_inv * logvar.exp() - var_inv * (mean - mu).pow(
        2))  # + latent_dim * torch.log(torch.Tensor([1/std_inv]).cuda())
    return kl_div


def wasserstein_distance(mu, logvar, mean):
    unit_var = torch.ones(logvar.shape[1]).float().cuda()
    distance = torch.sqrt(
        torch.sum((mu - mean) ** 2, dim=1) + torch.sum((torch.sqrt(logvar.exp()) - torch.sqrt(unit_var)) ** 2, dim=1))
    distance = distance.sum()
    return distance


def wasserstein_distance_vector(mu, logvar, mean):
    unit_var = torch.ones(logvar.shape[1]).float().cuda()
    distance = torch.sqrt(
        torch.sum((mu - mean) ** 2, dim=1) + torch.sum((torch.sqrt(logvar.exp()) - torch.sqrt(unit_var)) ** 2, dim=1))
    return distance


def kl_with_gaussian_unit_std(mu, logvar, mean):
    kl_div = -0.5 * torch.sum(1 + logvar - logvar.exp() - (mean - mu).pow(2))
    return kl_div


def wasserstein_distance_mean_only(mu, mean):
    distance = torch.sqrt(torch.sum((mu - mean) ** 2, dim=1))
    distance = distance.sum()
    return distance


# Constrastive Semantic Alignment Loss
def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)  # L2 norm pairwise
    loss = class_eq * dist.pow(2)  # Included distance only for similar classes, different domain
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(
        2)  # Include this part only for different classes, different domain
    return loss.mean()


# MMD (Maximum Mean Discrepancy) loss
# Source : https://github.com/tmac1997/DDC-transfer-learning/blob/master/mmd.py
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


# Reference: DCORAL: Correlation Alignment for Deep Domain Adaptation, ECCV-16.
def coral_loss(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss


def d_sne_loss(z_src, y_src, z_tgt, y_tgt, margin=1):
    z_src_rpt = z_src.unsqueeze(dim=0)
    z_tgt_rpt = z_tgt.unsqueeze(dim=1)

    dists = torch.sum((z_src_rpt - z_tgt_rpt) ** 2, dim=2)

    yt_rpt = y_tgt.unsqueeze(dim=1)
    ys_rpt = y_src.unsqueeze(dim=0)

    y_same = torch.eq(yt_rpt, ys_rpt).float()
    y_diff = torch.ne(yt_rpt, ys_rpt).float()

    intra_cls_dists = dists * y_same
    inter_cls_dists = dists * y_diff

    max_dists, _ = torch.max(dists, dim=1)
    revised_inter_cls_dists = torch.where(y_same.byte(), max_dists, inter_cls_dists)

    max_intra_cls_dist, _ = torch.max(intra_cls_dists, dim=1)
    min_inter_cls_dist, _ = torch.min(revised_inter_cls_dists, dim=1)

    loss = torch.relu(max_intra_cls_dist - min_inter_cls_dist + margin)

    return loss.mean()


def prototypical_loss_full(prototypes, z_val, y_val):
    '''
    Prototypical Loss Implemented
    :param z_train:
    :param y_train:
    :param z_val:
    :param y_val:
    :return:
    '''


    classes = torch.unique(y_val)
    n_classes = len(classes)
    n_query_real = None
    n_query_fake = None

    n_query_real = y_val.eq(classes[0].item()).sum().item()
    n_query_fake = y_val.eq(classes[1].item()).sum().item()
    n_query = min(n_query_real, n_query_fake)


    query_idx_real = y_val.eq(classes[0].item()).nonzero()[:n_query]
    query_idx_fake = y_val.eq(classes[1].item()).nonzero()
    r_perm = torch.randperm(query_idx_fake.shape[0])
    query_idx_fake = query_idx_fake[r_perm][:n_query]
    query_idxs = torch.stack((query_idx_real, query_idx_fake)).view(-1)
    query_samples = z_val[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)   # total_samples X 2 matrix (every sample distance with everyy prototype)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)  # 2 X Samples_per_class X 2

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().cuda()

    loss_val = - torch.gather(log_p_y, 2, target_inds.cuda()).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val