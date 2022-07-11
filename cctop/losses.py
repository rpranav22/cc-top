import torch.nn as nn
import torch.nn.functional as F
import torch

class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(KCL,self).__init__()
        self.kld = KLDiv()
        self.margin = margin

    def forward(self, prob1, prob2, simi):

        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        # kld calculated over all possible permutations
        # prob1.shape = prob2.shape = (batch_size**, num_classes)
        kld = self.kld(prob1,prob2)

        # self-build hinge loss, same time + result as pt's F.hinge_embedding_loss
        # important to filter out 0s as we have sparse matrices in the CM case
        kld_plus1 = kld[simi == 1]
        kld_minus1 = torch.clamp(input=self.margin-kld[simi == -1], min=0.0)
        output = torch.mean(torch.cat((kld_plus1, kld_minus1)))
        
        # # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        # output = torch.mean(F.hinge_embedding_loss(input=kld, target=simi, margin=self.margin, reduction='none')[simi != 0])
        return output

eps = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine using 1e-35 (float32).

class KLDiv(nn.Module):
    # Calculate KL-Divergence

    def forward(self, predict, target):
       assert predict.ndimension()==2,'Input dimension must be 2'
       target = target.detach()
       # KL(T||I) = \sum T(logT-logI)
       predict += eps
       target += eps
       logI = predict.log()
       logT = target.log()
       TlogTdI = target * (logT - logI)
       kld = TlogTdI.sum(1)
       return kld


class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(MCL.eps).log_()

        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        return torch.mean(neglogP[simi != 0])
