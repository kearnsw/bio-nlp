"""
HTML Taggers

@author: Will Kearns
"""
from collections import Counter
from argparse import ArgumentParser
from spacy import displacy
import os
import json
import ast
import sys
from bionlp.annotate.MetaMap import run_metamap
import string
from typing import Dict, List


class Token:
    def __init__(self, start_char: int, end_char: int, surface_form: str, label: str):
        self.startIndex = start_char
        self.endIndex = end_char
        self.word = surface_form
        self.type = label
        self.startTime = None
        self.endTime = None


class Doc:
    def __init__(self, _id: int):
        self.id = _id
        self.tokens: List[Token] = []


class NLU:
    """
    Class: Annotator
    Description: Generates and serves HTML annotations from an annotation file

    Must override the parse method for specific doc type
    """

    def __init__(self):
        self.entities: List[Token] = []
        self.text = None
        self.options = None
        self.title = None
        self.timestamps: List = []
        self.whitelist = ["neop", "dsyn", "vita", "virs", "phsu", "phsf", "clnd", "bpoc", "anab", "cell"]
        self.semtypes = {}
        self.load_semtype_dict()
        self.questions = []

    def parse(self, text, mm_out):
        self.text = text
        for doc in mm_out["AllDocuments"]:
            doc = doc["Document"]
            for utt in doc["Utterances"]:
                for phrase in utt["Phrases"]:
                    tokens = self.tokenize(phrase)
                    if phrase["Mappings"] is not []:
                        for mapping in phrase["Mappings"]:
                            for cand in mapping["MappingCandidates"]:
                                symtypes = cand["SemTypes"]
                                for _type in symtypes:
                                    if _type in self.whitelist:
                                        term = self.semtypes[_type]
                                        start = int(cand["ConceptPIs"][0]["StartPos"])
                                        for token in tokens:
                                            if token.startIndex == start:
                                                token.type = term.upper()
                            break
                    self.entities += tokens
        return self.entities

    @staticmethod
    def tokenize(phrase):
        entities = []  # Used to store all tokens temporarily until we can check UMLS mappings
        tokens = phrase["SyntaxUnits"]
        phrase_start = int(phrase["PhraseStartPos"])
        for token in tokens:
            phrase_text = token["InputMatch"]
            phrase_end = int(phrase_start) + len(phrase_text)
            entities.append(Token(start_char=phrase_start, end_char=phrase_end, surface_form=phrase_text, label=None))
            phrase_start = phrase_end + 1  # Add one for space
        return entities

    def add_timestamps(self):
        offset = 0          # Keep track of punctuations to offset the index to match up with GC Speech

        for idx, ent in enumerate(self.entities):
            text = ent.word

            if text in string.punctuation:
                ent.type = "PUNC"
                offset += 1
            start = idx - offset
            if "'" in text:
                offset += 1
                end = start + len(text.split()) - 2
            else:
                end = start + len(text.split()) - 1

            offset += start - end       # Update offset for length of token

            ent.startTime = self.timestamps[start][1]
            ent.endTime = self.timestamps[end][2]

    def parse_questions(self):
        for idx, token in enumerate(self.entities):
            if token.word == "?":
                end_idx = idx
                start_idx = idx
                while start_idx >= 0:
                    start_idx -= 1
                    if self.entities[start_idx].type == "PUNC":
                        start_idx += 1
                        break

                if end_idx != start_idx:
                    text = " ".join([entity.word for entity in self.entities[start_idx:end_idx+1]]).capitalize()
                    self.questions.append({"startIndex": start_idx, "endIndex": end_idx, "text": text})


    def load_semtype_dict(self):
        dir = os.path.dirname(__file__)
        with open(os.path.join(dir, 'SemanticTypes_2013AA.txt'), "r") as input_file:
            for line in input_file.readlines():
                line = line.split("|")
                self.semtypes[line[0]] = line[2].strip()


def main():
    """

    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('--pipe', action='store_true', help='take input from stdin')
    parser.add_argument('--text', type=str, help='raw text to be annotated')
    parser.add_argument('--file', type=str, help='file containing the raw text')
    parser.add_argument('--annotations', type=str, help='annotation file in metamap (mm) or (mml) format ')
    parser.add_argument('--format', type=str, default="mm", help='use metamap (mm) or metamap lite (mml)')
    parser.add_argument('--output_format', type=str, default="json", help='html or json output')
    parser.add_argument('--timestamps', type=str, default=None,
                        help='list of tab delimited time stamp (word,start,end')
    args = parser.parse_args()

    # Get text input
    if args.pipe:
        raw_text = sys.stdin.read()
    else:
        raw_text = args.text

    # Get annotations
    try:
        with open(args.annotations, "r") as f:
            annotations = f.read()
    except:
        annotations = run_metamap(raw_text)
        with open(args.annotations, "wb") as f:
            f.write(annotations)

    # Create parser
    if args.format == "mm":
        parser = NLU()
        annotations = annotations.split("\n")
        if isinstance(annotations, list):
            annotations = json.loads("\n".join(annotations[1:]))  # ignore header

    # Load timestamps
    if args.timestamps:
        parser.timestamps = []
        for timestamp in ast.literal_eval(args.timestamps):
            timestamp = timestamp.split("\t")
            parser.timestamps.append((timestamp[0], timestamp[1], timestamp[2]))


    # Parse the data
    ents = parser.parse(raw_text, annotations)
    ents = [ent.__dict__ for ent in ents]
    parser.add_timestamps()
    parser.parse_questions()
    sys.stdout.write(json.dumps({"tokens": ents, "questions": parser.questions}))
    sys.stdout.flush()


if __name__ == "__main__":
    """
    sys.argv.append("--text")
    sys.argv.append("Hello, what is my current level of my white blood cell? How much white blood cell she'd like"
                    " assumed to keep myself healthy. White blood cell, okay.")
    sys.argv.append("--timestamps")
    sys.argv.append(["Hello,	0.8	1.3",
                     "what	1.6	2.1",
                    "is	2.1	2.2",
                     "my	2.2	2.2",
                     "current	2.2	3.8",
                     "level	3.8	4.2",
                     "of	4.2	4.3",
                     "my	4.3	4.6",
                     "white	4.6	5.3",
                     "blood	5.3	5.4",
                     "cell?	5.4	5.9",
                     "How	7.0	8.1",
                     "much	8.1	8.3",
                     "white	8.3	9.4",
                     "blood	9.4	9.6",
                     "cell	9.6	9.8",
                     "she'd	9.8	10.1",
                     "like	10.1	10.3",
                     "assumed	10.3	10.6",
                     "to	10.6	11.1",
                     "keep	11.1	11.4",
                     "myself	11.4	11.7",
                     "healthy.	11.7	12.2",
                     "White	15.1	15.6",
                     "blood	15.6	15.9",
                     "cell,	15.9	16.1",
                     "okay.	16.1	17.0"])
    sys.argv.append("--annotations")
    sys.argv.append("annotations.txt")
    """
    main()

