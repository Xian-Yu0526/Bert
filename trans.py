import os

VOCAB_FILE = "vocab.txt"
with open(VOCAB_FILE, "r+", encoding="utf-8") as f:
    tokens = f.read().split()

for i in [0 ,276, 1135 ,1529, 1 ,826 ,2473, 910, 3115, 2808, 590 ,
          42 ,1, 2815 ,1856 ,2244 ,1118 ,1 ,3 ,2885 ,1533, 36 ,3157 ,1804 ,844 ,1441, 1210]:
    print(tokens[i], end=" ")
