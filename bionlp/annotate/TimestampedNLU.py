"""
Natural Language Understanding

@Copyright Will Kearns
"""


from argparse import ArgumentParser
import os
import sys
import string
import json
from bionlp.annotate.MetaMap import run_metamap
from typing import Dict, List, Tuple


class Token:
    def __init__(self, start_char: int, end_char: int, surface_form: str, label: str):
        """
        A token consists of one or more words and holds meta-information about the token

        :param start_char: character index of the first character in the token
        :param end_char: character index of the last character in the token
        :param surface_form: the text of the token
        :param label: semantic type of the token
        """
        self.startIndex: int = start_char
        self.endIndex: int = end_char
        self.word: str = surface_form
        self.type: str = label
        self.startTime: int = None
        self.endTime: int = None
        self.cui: str = None


class NLU:
    def __init__(self):
        """
        Class: Natural Language Understanding Component
        Description: Connect time-indexed speech recognition with MetaMap named entity recognition
        """
        self.doc: List[Token] = []
        self.text: str = None
        self.options: Dict = None
        self.timestamps: List = [Tuple]
        self.code2semtype: Dict[str, str] = self.load_semtype_dict()
        self.entities: List[Token] = []
        self.title = None
        self.whitelist = ["antb", "neop", "dsyn", "vita", "virs", "phsu", "phsf", "clnd", "bpoc", "anab", "cell"]
        self.semtypes = {}
        self.load_semtype_dict()
        self.questions = []
        self.index = 0

    def parse(self, text, mm_out):
        """
        Parses Metamap output
        :param text: Input Text
        :param mm_out: Metamap output
        :return: Tokens
        """
        self.text = text
        for doc in mm_out["AllDocuments"]:
            doc = doc["Document"]
            for utt in doc["Utterances"]:
                # Loop over all phrases in the document
                for phrase in utt["Phrases"]:
                    tokens = self.tokenize(phrase)
                    # Check for mapping to UMLS terminologies
                    if phrase["Mappings"] is not []:
                        for mapping in phrase["Mappings"]:
                            for cand in mapping["MappingCandidates"]:
                                symtypes = cand["SemTypes"]
                                for _type in symtypes:
                                    if _type in self.whitelist:
                                        term = self.code2semtype[_type]
                                        # start = int(cand["ConceptPIs"][0]["StartPos"])
                                        for token in tokens:
                                            for word in token.word.lower().split():
                                                print(word)
                                                print(cand["MatchedWords"])
                                                if word in cand["MatchedWords"]:
                                                    token.type = term.upper()
                            if any([token.type is not None for token in tokens]):
                                break
                    self.doc += tokens
        return self.doc

    def tokenize(self, phrase):
        entities = []  # Used to store all tokens temporarily until we can check UMLS mappings
        tokens = phrase["SyntaxUnits"]
        for token in tokens:
            phrase_text = token["InputMatch"]
            if phrase_text in [",", ":", ";", "!", "?", "."]:
                self.index -= 1
            phrase_end = int(self.index) + len(phrase_text) - 1
            entities.append(Token(start_char=self.index, end_char=phrase_end, surface_form=phrase_text, label=None))
            self.index = phrase_end + 2
        return entities

    def add_timestamps(self):
        offset = 0          # Keep track of punctuations to offset the index to match up with GC Speech

        for idx, ent in enumerate(self.doc):
            text = ent.word

            if text in string.punctuation:
                ent.type = "PUNC"
                offset += 1
            start = idx - offset

            if " '" in text:
                offset += 1
                end = start + len(text.split()) - 2
            else:
                end = start + len(text.split()) - 1

            offset += start - end       # Update offset for length of token
            ent.startTime = float(self.timestamps[start][1])
            ent.endTime = float(self.timestamps[end][2])

    def parse_questions(self):
        for idx, token in enumerate(self.doc):
            if token.word == "?":
                end_idx = idx
                start_idx = idx
                while start_idx >= 0:
                    start_idx -= 1
                    if self.doc[start_idx].type == "PUNC":
                        start_idx += 1
                        break

                if end_idx != start_idx:
                    text = " ".join([entity.word for entity in self.doc[start_idx:end_idx + 1]]).capitalize()
                    self.questions.append({"startTokenIndex": start_idx, "endTokenIndex": end_idx, "text": text})
        return self.questions

    @staticmethod
    def load_semtype_dict():
        _dir = os.path.dirname(__file__)
        semtypes = {}
        with open(os.path.join(_dir, 'SemanticTypes_2013AA.txt'), "r") as input_file:
            for line in input_file.readlines():
                line = line.split("|")
                semtypes[line[0]] = line[2].strip()
        return semtypes


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
    annotations = run_metamap(raw_text)

    # Create parser
    if args.format == "mm":
        parser = NLU()
        annotations = annotations.decode().split("\n")

        if isinstance(annotations, list):
            annotations = json.loads("\n".join(annotations[1:]))  # ignore header

    # Load timestamps
    parser.timestamps = []
    if args.timestamps:
        timestamp = args.timestamps.split("|,")
        for ts in timestamp:
            parser.timestamps.append(ts.split("\t"))

    # Parse the data
    doc = parser.parse(raw_text, annotations)
    doc = [token.__dict__ for token in doc]
    if args.timestamps:
        parser.add_timestamps()
    qs = parser.parse_questions()
    materials = [
                    {
                        "startTokenIndex": 0,
                        "endTokenIndex": 3,
                        "title": "Side effects of not washing your hands",
                        "url": "https://medlineplus.gov/asthma.html?utm_medium=service&utm_source=mplusconnect",
                        "source": "Blah.org",
                        "summary": "<p>Asthma is a chronic disease that affects your airways. Your airways are tubes that carry air in and out of your lungs. If you have asthma, the inside walls of your airways become sore and swollen. That makes them very sensitive, and they may react strongly to things that you are <a href=\"https://medlineplus.gov/allergy.html?utm_medium=service&amp;utm_source=mplusconnect\">allergic</a> to or find irritating. When your airways react, they get narrower and your lungs get less air.</p>\n\n<p>Symptoms of asthma include</p>\n\n<ul>\n<li>Wheezing</li>\n<li>Coughing, especially early in the morning or at night</li>\n<li>Chest tightness</li>\n<li>Shortness of breath</li>\n</ul>\n\n<p>Not all people who have asthma have these symptoms. Having these symptoms doesn't always mean that you have asthma. Your doctor will diagnose asthma based on lung function tests, your medical history, and a physical exam. You may also have allergy tests.</p>\n\n<p>When your asthma symptoms become worse than usual, it's called an asthma attack. Severe asthma attacks may require emergency care, and they can be fatal.</p>\n\n<p>Asthma is treated with two kinds of medicines: quick-relief medicines to stop asthma symptoms and long-term control medicines to prevent symptoms. </p>\n\n<p class=\"NLMattribution\">  NIH: National Heart, Lung, and Blood Institute</p>\n<p class=\"NLMrelatedLinks\"><ul><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000487.htm?utm_medium=service&amp;utm_source=mplusconnect\">Allergies, asthma, and dust</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000488.htm?utm_medium=service&amp;utm_source=mplusconnect\">Allergies, asthma, and molds</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000489.htm?utm_medium=service&amp;utm_source=mplusconnect\">Allergies, asthma, and pollen</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/article/000141.htm?utm_medium=service&amp;utm_source=mplusconnect\">Asthma</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000005.htm?utm_medium=service&amp;utm_source=mplusconnect\">Asthma - control drugs</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000008.htm?utm_medium=service&amp;utm_source=mplusconnect\">Asthma - quick-relief drugs</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000036.htm?utm_medium=service&amp;utm_source=mplusconnect\">Exercise-induced asthma</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000053.htm?utm_medium=service&amp;utm_source=mplusconnect\">How to breathe when you are short of breath</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000006.htm?utm_medium=service&amp;utm_source=mplusconnect\">How to use a nebulizer</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000041.htm?utm_medium=service&amp;utm_source=mplusconnect\">How to use an inhaler - no spacer</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000042.htm?utm_medium=service&amp;utm_source=mplusconnect\">How to use an inhaler - with spacer</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/article/003853.htm?utm_medium=service&amp;utm_source=mplusconnect\">Pulmonary function tests</a> (Medical Encyclopedia)\n        </li><li><a href=\"https://medlineplus.gov/ency/patientinstructions/000062.htm?utm_medium=service&amp;utm_source=mplusconnect\">Signs of an asthma attack</a> (Medical Encyclopedia)\n        </li></ul></p>"
                    }
                ]

    sys.stdout.write(json.dumps({"tokens": doc, "questions": qs, "educationalMaterials": materials}))
    sys.stdout.flush()


if __name__ == "__main__":
    text = "you dont know what cancer is until you no longer have it and you think you do"
    print(run_metamap(text))
    main()

