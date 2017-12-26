from torch.utils.data import Dataset
from bionlp.utils.BratParser import BratParser
from bs4 import BeautifulSoup
import os


class GARD(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data, self.labels = self.load(path)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            xml_data = f.read()
        questions = BeautifulSoup(xml_data, "xml").find_all("SubQuestion")
        data = [q.text for q in questions]
        labels = [q["qt"] for q in questions]
        return data, labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class CADEC(Dataset):
    def __init__(self, text_dir, ann_dir):
        super().__init__()
        self.data, self.labels = self.load(text_dir, ann_dir)

    @staticmethod
    def load(text_dir, ann_dir):

        parser = BratParser()
        text = {}
        bio = {}
        for filename in os.listdir(text_dir):
            txt_path = text_dir + filename
            ann_path = ann_dir + filename.replace(".txt", ".ann")
            bio[filename.replace(".txt", "")] = parser.parse_to_bio(txt_path, ann_path)
            text[filename.replace(".txt", "")] = parser.text

        docs = []
        labels = []
        misaligned_count = 0
        for idx, fn in enumerate(bio):
            raw_text = strip_punctuation(text[fn]).lower().split()
            bio_seq = bio[fn]

            # Don't include files with no text (why are these even in the data set??)
            if not raw_text:
                continue
            if len(raw_text) != len(bio_seq):
                misaligned_count += 1
                continue
            docs.append(raw_text)
            labels.append(bio_seq)

        print("Misaligned documents: {0}".format(misaligned_count))
        return docs, labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def strip_punctuation(s):
    for c in punctuation:
        s = s.replace(c, " ")
    return s


punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '.', ':', ';',
               '=', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '/', '.', ',']


if __name__ == "__main__":
    data = GARD("/home/will/projects/GARD/data/GARD_qdecomp.master.03.qtd.xml")
    print(data.__getitem__(0))

