import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class CULoss(nn.Module):
    def __init__(self, conf, beta, non=False):
        super(CULoss, self).__init__()
        self.loss = nn.SoftMarginLoss()
        self.beta = beta
        self.non = non
        if conf.mean() > 0.5:
            self.UP = True
        else:
            self.UP = False

    def forward(self, conf, labeled, unlabeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))
        
        if self.UP:
            #conf_risk = torch.mean((1-conf) * (y_conf_neg - y_conf_pos) + (1 - self.beta) * y_conf_pos)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, torch.ones(unlabeled.shape).to(device)))
            neg_risk = torch.mean((1 - conf) * y_conf_neg)
            pos_risk = torch.mean((conf - self.beta) * y_conf_pos) + unlabeled_risk
        else:
            #conf_risk = torch.mean(conf * (y_conf_pos - y_conf_neg) + (1 - self.beta) * y_conf_neg)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, -torch.ones(unlabeled.shape).to(device)))
            pos_risk = torch.mean(conf * y_conf_pos)
            neg_risk = torch.mean((1 - self.beta - conf) * y_conf_neg) + unlabeled_risk
        if self.non:
            objective = torch.clamp(neg_risk, min=0) + torch.clamp(pos_risk, min=0)
        else:
            objective = neg_risk + pos_risk
        return objective


class PNLoss(nn.Module):
    def __init__(self):
        super(PNLoss, self).__init__()
        self.loss = nn.SoftMarginLoss()

    def forward(self, conf, labeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        objective = torch.mean(conf * y_conf_pos + (1 - conf) * y_conf_neg)
        return objective
