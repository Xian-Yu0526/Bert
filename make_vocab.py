import os

DIR_PATHS = [r".\data"]
VOCAB_FILE = "vocab.txt"

words = set()

for DIR_PATH in DIR_PATHS:
    for i, filename in enumerate(os.listdir(DIR_PATH)):
        f_path = os.path.join(DIR_PATH, filename)
        print(f_path)
        with open(f_path, "r+", encoding="utf-8") as f:
            w = f.read(1)
            print(w)
            while w:
                if w == '\n' or w == '\r' or w == ' ':
                    # words.add('[SEP]')
                    pass
                else:
                    words.add(w)
                w = f.read(1)

with open(VOCAB_FILE, "w+", encoding="utf-8") as f:
    f.write("[START] [SEQ] [UNK] [PAD] [END]")
    f.write(" ".join(words))
    f.flush()
