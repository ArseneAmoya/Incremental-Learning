import torch
import torch.nn as nn
import torch.nn.functional as F

class Similarity_preserving(nn.Module):
    def __init__(self, num_class=0, device = "cuda:0"):
        super(Similarity_preserving, self).__init__()
        self.use_gpu = True
        self.T = 4
        self.device = device

    def forward(self, FeaT, FeaS):

        batch_size = FeaT.size()[0]
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)
        
        FeaS  = F.normalize(FeaS.to(self.device),p=2, dim=1)
        FeaT = F.normalize(FeaT.to(self.device), p=2, dim=1)

        # calculate the similar matrix
        Sim_T = torch.mm(FeaT, FeaT.t())
        Sim_S = torch.mm(FeaS, FeaS.t())
        # print("range of Sim_T", Sim_T.min().item(), Sim_T.max().item())
        # print("range of Sim_S", Sim_S.min().item(), Sim_S.max().item())


        # kl divergence
        p_s = F.log_softmax(Sim_S / self.T, dim=1)
        p_t = F.softmax(Sim_T / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / Sim_S.shape[0]
        # print("loss kl div", loss.item())
        return loss