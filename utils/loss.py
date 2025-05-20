import torch


def bpr_loss(u, pos, neg):
    pos_score = torch.sum(u * pos, dim=1)
    neg_score = torch.sum(u * neg, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))
    return loss

def l2_reg_loss(reg, *args):
    return reg * sum(torch.norm(x) ** 2 for x in args) / len(args)
