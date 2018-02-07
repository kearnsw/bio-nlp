"""
HTML Taggers

@author: Will Kearns
"""
from bionlp.core.LinkedList import LinkedList
from collections import Counter
from argparse import ArgumentParser, Namespace
from spacy import displacy
import os
import json
import ast
import sys
from bionlp.annotate.MetaMap import run_metamap


from typing import Dict, List


class HTMLAnnotator(object):
    """
    Class: Annotator
    Description: Generates and serves HTML annotations from an annotation file
    
    Must override the parse method for specific doc type 
    """

    def __init__(self):
        self.entities = []
        self.annotations = []
        self.text = None
        self.char_ll = LinkedList()
        self.options = None
        self.title = None
        self.timestamps = None

    def parse(self, text, ann):
        pass

    def define_colors(self, palette):
        """
        Set color options for renderer
        
        :param palette: (list) Set of colors to map entities to
        :return: None
        """
        # Define colors
        colors = {}
        i = 0
        for ent in self.annotations[-1]["ents"]:
            if ent["label"] not in colors:
                colors[ent["label"]] = palette[i % len(palette)]
                i += 1

        self.options = {'ents': list(colors.keys()), 'colors': colors}

    def serve(self):
        """
        Run displacy server of parse
        :return: None
        """
        return displacy.serve(self.annotations, style='ent', manual=True, options=self.options)

    def render(self):
        """
        Write annotations as an HTML file
        :return: 
        """
        return displacy.render(self.annotations, style='ent', manual=True, options=self.options)

    def ingest_text(self, text):
        """
        Store text and create a linked list of characters for inserting XML tags
        :param text: 
        :return: a linked list of characters
        """
        self.text = text
        # Convert text to char linked list
        self.char_ll.from_string(text)

        return self.char_ll
    
    def prep_entities(self) -> List:
        """
        Prepare output from metamap for use in API
        1) Adds timestamps to words
        2) Merges entities
        3) Calculate counts
        4) Format as JSON Array
        :return: JSON array of with keys display name, type, count, and timestamp.
        """
        indices = {}
        entities = [entity["label"] for entity in self.entities]
        if self.timestamps:
            for entity in self.entities:
                indices[entity["label"]] = self.find_time_of_string(entity["surface"])

        counts = Counter(entities)
        res = []
        for entity, count in counts.items():
            display_name, cui, semtype = entity.split(":")
            if self.timestamps:
                res.append({"display_name": display_name, "count": count, "type": semtype, "timestamp": indices[entity]})
            else:
                res.append({"display_name": display_name, "count": count, "type": semtype})
        return res

    def find_time_of_string(self, s: str) -> List:
        """
        Check list of (word, start) tuples for matching strings
        :param s: the original surface form of the entity to be matched
        :return: a list of start times in ms 
        """
        tokens = s.split()
        starttimes = []
        for i, timestamp in enumerate(self.timestamps):
            if tokens[0] == timestamp[0]:
                start = timestamp[1]
                if len(tokens) > 1:
                    for j, token in enumerate(tokens[1:]):
                        if token == self.timestamps[i + j + 1]:
                            if j == len(tokens[1:]):
                                starttimes.append(start)
                            continue
                        else:
                            break
                else:
                    starttimes.append(start)
        return starttimes


class MetaMapLiteAnnotator(HTMLAnnotator):
    """
    Class: MetaMapLiteAnnotator
    Description: MetaMapLiteAnnotator uses the output of MetaMapLite to label the reference text entities with HTML tags
    """

    def __init__(self):
        super().__init__()
        self.title = "MetaMap Lite"

    def parse(self, text, mm_out):
        # Store text for later use
        self.ingest_text(text)

        for row in mm_out:
            row = row.split("|")
            # Make sure all metadata is available
            if len(row) >= 8:
                term = row[2] + ":" + row[3]
                span = row[7]
                start, offset = [int(term) for term in span.split(":")]
                end = start + offset
                self.entities.append({'start': start, 'end': end, 'label': term.upper()})

        self.annotations.append({'text': text, 'ents': self.entities, 'title': "MetaMap Lite"})

        return json.dumps(self.annotations)


class MetaMapAnnotator(HTMLAnnotator):
    """
    Class: MetaMapAnnotator 
    Description: MetaMapAnnotator parses the JSON output of MetaMap to label the entities in a reference text with HTML 
                 tags
    """

    def __init__(self):
        super().__init__()
        self.title = "MetaMap"
        self.whitelist = ["neop", "dsyn", "vita", "virs", "phsu", "phsf", "clnd", "bpoc", "anab"]
        self.symtypes = {}
        self.load_semtype_dict()

    def parse(self, text, mm_out):
        self.ingest_text(text)
        for doc in mm_out["AllDocuments"]:
            doc = doc["Document"]
            for utt in doc["Utterances"]:
                for phrase in utt["Phrases"]:
                    phrase_start = phrase["PhraseStartPos"]
                    phrase_end = int(phrase_start) + int(phrase["PhraseLength"])
                    cuis = []
                    for mapping in phrase["Mappings"]:
                        for cand in mapping["MappingCandidates"]:
                            score = cand["CandidateScore"]
                            cui = cand["CandidateCUI"]
                            symtypes = cand["SemTypes"]
                            if cui in cuis:
                                break
                            cuis.append(cui)
                            term = cand["CandidatePreferred"]
                            surface = cand["MatchedWords"]
                            term = " ".join(surface)
                            term += ":" + cui
                            for _type in symtypes:
                                if _type in self.whitelist:
                                    term += ":" + self.symtypes[_type]

                                    ## Possible bug here need to adjust for discontinuous entities
                                    loc = cand["ConceptPIs"][0]
                                    start = int(loc["StartPos"])
                                    end = start + int(loc["Length"])
                                    self.entities.append({'start': start, 'end': end, 'label': term.upper(),
                                                          'surface': " ".join(surface)})
                        break
        self.annotations.append({"text": self.text, "ents": self.entities, "title": self.title})

        return self.annotations

    def load_semtype_dict(self):
        dir = os.path.dirname(__file__)
        with open(os.path.join(dir, 'SemanticTypes_2013AA.txt'), "r") as input_file:
            for line in input_file.readlines():
                line = line.split("|")
                self.symtypes[line[0]] = line[2].strip()


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
    parser.add_argument('--output_format', type=str, default="html", help='html or json output')
    parser.add_argument('--timestamps', type=str, default=None,
                        help='list of tab delimited time stamp (word,start,end')
    args = parser.parse_args()

    # Get text input
    if args.pipe:
        raw_text = sys.stdin.read()
    elif args.text:
        raw_text = args.text
    else:
        with open(args.text, "r") as f:
            raw_text = f.read()

    # Get annotations
    try:
        with open(args.annotations, "r") as f:
            annotations = f.read()
    except:
        annotations = run_metamap(raw_text)

    # Create parser
    if args.format == "mm":
        parser = MetaMapAnnotator()
        if isinstance(annotations, list):
            annotations = json.loads(annotations[1:])  # ignore header
    else:
        parser = MetaMapLiteAnnotator()

    # Load timestamps
    if args.timestamps:
        parser.timestamps = []
        for timestamp in ast.literal_eval(args.timestamps):
            timestamp = timestamp.split("\t")
            parser.timestamps.append((timestamp[0], timestamp[1]))

    # Parse the data
    parser.parse(raw_text, annotations)

    # Output data as html or as a json object
    if args.output_format == "html":
        parser.define_colors(['#9b38bd', '#34c3b9', '#abd8d8', '#c2d54a', '#e0eaa9'])
        sys.stdout.write(parser.render())
        sys.stdout.flush()
    else:
        sys.stdout.write(json.dumps(parser.prep_entities()))
        sys.stdout.flush()


if __name__ == "__main__":
    main()

