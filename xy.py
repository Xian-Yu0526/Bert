import torch
VOCAB_FILE = "vocab.txt"

with open(VOCAB_FILE, "r+", encoding="utf-8") as f:
    tokens = f.read().split()

a=input()
b=[]
c=[]
z=len(a)
for i in range(z):

    b.append(tokens.index(a[i]))
    c.append(i)
print(torch.tensor([b]))
print(z)
print(torch.tensor([c]))