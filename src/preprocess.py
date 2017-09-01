import pandas as pd
import os
import re
from LinkedList import LinkedList

punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '.', ':', ';',
               '=', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '/', '.', ',']


def strip_punctuation(s):
    for c in punctuation:
        s = s.replace(c, " ")
    return s


def load_training_data(text_dir, ann_dir):

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
    for idx, fn in enumerate(bio):
        raw_text = strip_punctuation(text[fn]).split()
        bio_seq = bio[fn]

        # Don't include files with no text (why are these even in the data set??)
        if not raw_text:
            continue
        docs.append(raw_text)
        labels.append(bio_seq)

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

"""
    if "indexed_data.pkl" not in os.listdir("."):
        # Load training data and split into training and dev
        docs, labels = load_training_data(args.text, args.ann)
        split_idx = 10
        x_train = docs[:split_idx]
        y_train = labels[:split_idx]
        x_dev = docs[split_idx:]
        y_dev = docs[split_idx:]

        # Load the word vectors and index the vocabulary of the embeddings and annotation tags
        embeddings, word2idx = load_embeddings(args.embedding, args.emb_dim)
        tag2idx = idx_tags(y_train)

        # Convert data into sequence using indices from embeddings
        x_train = [text2seq(sentence, word2idx, pytorch=True) for sentence in x_train]
        y_train = [tags2idx(tag_seq, tag2idx, pytorch=True) for tag_seq in y_train]

        with open("indexed_data.pkl", "wb") as f:
            pkl.dump({"x": x_train, "y": y_train, "embeddings": embeddings, "word2idx": word2idx, "tag2idx": tag2idx}, f)

    else:
        with open("indexed_data.pkl", "rb") as f:
            data = pkl.load(f)
            embeddings = data["embeddings"]
            word2idx = data["word2idx"]
            tag2idx = data["tag2idx"]
            x_train = data["x"]
            y_train = data["y"]
"""