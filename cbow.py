import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self,voc_num,voc_dim):
        super().__init__()
        self.codebook=torch.Embedding(voc_num,voc_dim)
        self.Leaner_1=nn.Linear(voc_num,voc_num,bias=False)
    def forward(self,x1,x2,x4,x5):
        v1=self.codebook(x1)
        v2 = self.codebook(x2)
        v4 = self.codebook(x4)
        v5 = self.codebook(x5)

        y1=self.Leaner_1(v1)
        y2 = self.Leaner_1(v2)
        y4 = self.Leaner_1(v4)
        y5 = self.Leaner_1(v5)

        return y1+y2+y4+y5
    def getLoss(self,x3,y3):
        v3=self.codebook(x3)

        return torch.mean((y3-v3)**2)