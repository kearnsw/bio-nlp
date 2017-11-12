"""
HTML Taggers

@author: Will Kearns
"""
from bionlp.core.LinkedList import LinkedList
from argparse import ArgumentParser
from spacy import displacy
import json


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

        return self.annotations


class MetaMapAnnotator(HTMLAnnotator):
    """
    Class: MetaMapAnnotator 
    Description: MetaMapAnnotator parses the JSON output of MetaMap to label the entities in a reference text with HTML 
                 tags
    """

    def __init__(self):
        super().__init__()
        self.title = "MetaMap"
        self.whitelist = ["dsyn", "vita", "virs", "phsu", "phsf", "clnd", "bpoc", "anab"]

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
                            term += ":" + cui
                            for _type in symtypes:
                                if _type in self.whitelist:
                                    term += ":" + _type

                                    ## Possible bug here need to adjust for discontinuous entities
                                    loc = cand["ConceptPIs"][0]
                                    start = int(loc["StartPos"])
                                    end = start + int(loc["Length"])
                                    self.entities.append({'start': start, 'end': end, 'label': term.upper()})

        self.annotations.append({'text': self.text, 'ents': self.entities, 'title': self.title})

        return self.annotations


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--pipe', action='store_true', help='take input from stdin')
    parser.add_argument('--text', type=str, help='raw text to be annotated')
    parser.add_argument('--annotations', type=str, help='metamap annotations')
    parser.add_argument('--version', type=str, default="mml", help='use metamap (mm) or metamap lite (mml)')
    args = parser.parse_args()

    if args.pipe:
        import sys
        from bionlp.annotate.MetaMap import run_metamap
        raw_text = sys.stdin.read()
        annotations = run_metamap(raw_text)
        parser = MetaMapAnnotator()
    elif args.text:
        import sys
        from bionlp.annotate.MetaMap import run_metamap
        raw_text = args.text
        annotations = run_metamap(raw_text)
        parser = MetaMapAnnotator()
    else:
        with open(args.text, "r") as f:
            raw_text = f.read()

        if args.version == "mm":
            parser = MetaMapAnnotator()
            with open(args.annotations, "r") as f:
                header = f.readline()
                data = f.read()
                annotations = json.loads(data)
        else:
            parser = MetaMapLiteAnnotator()
            with open(args.annotations, "r") as f:
                annotations = f.readlines()

    parser.parse(raw_text, annotations)
    parser.define_colors(['#9b38bd', '#34c3b9', '#abd8d8', '#c2d54a', '#e0eaa9'])
    sys.stdout.write(parser.render())
    sys.stdout.flush()
