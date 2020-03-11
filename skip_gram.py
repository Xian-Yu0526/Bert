import torch
import torch.nn as nn

class Skip_Gram(nn.Module):
    def __init__(self,voc_num,voc_dim):
        super().__init__()
        self.codebook=torch.Embedding(voc_num,voc_dim)

        self.Leaner_1 = nn.Linear(voc_num, voc_num, bias=False)
        self.Leaner_2 = nn.Linear(voc_num, voc_num, bias=False)
        self.Leaner_4= nn.Linear(voc_num, voc_num, bias=False)
        self.Leaner_5 = nn.Linear(voc_num, voc_num, bias=False)
    def forward(self,x3):
        v3=self.codebook(x3)

        y1=self.Leaner_1(v3)
        y2 = self.Leaner_2(v3)
        y4 = self.Leaner_4(v3)
        y5 = self.Leaner_5(v3)
        return y1,y2,y4,y5
    def get_loss(self,x1,x2,x4,x5,y1,y2,y4,y5):
        v1=self.codebook(x1)
        v2=self.codebook(x2)
        v4=self.codebook(x4)
        v5=self.codebook(x5)

        return torch.mean(((y1-v1)**2+(y2-v2)**2+(y4-v4)**2+(y5-v5)**2))