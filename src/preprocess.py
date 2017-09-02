import pandas as pd
import os
import re
from LinkedList import LinkedList
from torch.utils.data import Dataset
punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '.', ':', ';',
               '=', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '/', '.', ',']


class CadecDataset(Dataset):

    def __init__(self, text_dir, ann_dir, transform=None):
        self.data = load_data(text_dir, ann_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels = self.data[idx]
        if self.transform:
            seq = self.tranform(text)

        return


def strip_punctuation(s):
    for c in punctuation:
        s = s.replace(c, " ")
    return s


def load_data(text_dir, ann_dir):

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


class BratParser:
    def __init__(self, text=None, ann=None):
        self.label = None
        self.text = text
        self.ann = ann

    def load(self, raw_text_path, annotation_path):

        with open(raw_text_path, "r", encoding="ISO-8859-1") as f:
            self.text = f.read()

        with open(annotation_path, "r", encoding="ISO-8859-1") as f:

            try:
                self.ann = pd.read_csv(f, names=["id", "span", "surface_form"],
                                       sep='\t', comment='#')
                self.ann["span"] = self.ann["span"].apply(lambda s: s.replace(";", " ").split())

            except pd.io.common.EmptyDataError:
                self.ann = pd.DataFrame()

        return {"text": self.text, "ann": self.ann}

    def parse_to_bio(self, raw_text_path, annotation_path):
        self.load(raw_text_path, annotation_path)
        return self.convert_to_bio_sequence()

    @staticmethod
    def string_to_linked_list(s):
        ll = LinkedList()
        for idx, char in enumerate(s):
            ll.insert(idx, char)
        return ll

    def convert_to_bio_sequence(self):
        ll = LinkedList()
        ll.from_string(self.text)

        for _, ann in self.ann.iterrows():
            self.label = ann["span"][0]
            bounds = [int(bound) for bound in ann["span"][1:]]

            # This code pairs the boundaries handling non-continuous chunks
            span_num = 0
            for start, end in zip(bounds[::2], bounds[1::2]):
                if span_num == 0:
                    ll.insert_tag_at_idx(start, " <B-{0}> ".format(self.label))
                    ll.insert_tag_at_idx(end, " <\\B-{0}> ".format(self.label))
                else:
                    ll.insert_tag_at_idx(start, " <I-{0}> ".format(self.label))
                    ll.insert_tag_at_idx(end, " <\\I-{0}> ".format(self.label))
                span_num += 1

        begin = False
        inside = False
        tagged_seq = []

        for token in strip_punctuation(ll.to_string()).split():
            # Check if there is an xml tag and change state accordingly
            # else add BIO tag to sequence
            if re.match(r"<B-\w+>", token):
                label = re.search("<B-(\w+)>", token).group(1)
                begin = True
            elif re.match(r"<\\B-\w+>", token):
                begin = False
                inside = False
            elif re.match(r"<I-\w+>", token):
                label = re.search("<I-(\w+)>", token).group(1)
                inside = True
            elif re.match(r"<\\I-\w+>", token):
                begin = False
                inside = False
            else:
                if begin:
                    begin = False
                    inside = True
                    tagged_seq.append("B-{0}".format(label))
                elif inside:
                    tagged_seq.append("I-{0}".format(label))
                else:
                    tagged_seq.append("O")

        return tagged_seq


if __name__ == "__main__":

    text_dir = os.sep.join([args["dir"], "text", ""])
    ann_dir = os.sep.join([args["dir"], "original", ""])

    parser = BratParser()
    text = {}
    bio = {}
    for filename in os.listdir(text_dir):
        txt_path = text_dir + filename
        ann_path = ann_dir + filename.replace(".txt", ".ann")
        bio[filename.replace(".txt", "")] = parser.parse_to_bio(txt_path, ann_path)
        text[filename.replace(".txt", "")] = parser.text

    for idx, fn in enumerate(bio):

        if idx >= 100:
            break
        else:
            print("===== {0} =====".format(fn))
            raw_text = strip_punctuation(text[fn]).split()
            bio_seq = bio[fn]

        training_data = zip(raw_text, bio_seq)
