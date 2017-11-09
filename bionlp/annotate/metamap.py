from bionlp.core.LinkedList import LinkedList
from argparse import ArgumentParser
from spacy import displacy


class MetaMapAnnotator(object):
    def __init__(self):
        self.entities = []
        self.annotations = []
        self.text = None
        self.options = None
        self.title = None

    def annotate(self, text, mm_out):

        # Store text for later use
        self.text = text

        # Convert text to char linked list
        ll = LinkedList()
        ll.from_string(text)

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

    def define_colors(self, palette):
        # Define colors
        colors = {}
        i = 0
        for ent in self.annotations[-1]["ents"]:
            if ent["label"] not in colors:
                colors[ent["label"]] = palette[i % len(palette)]
                i += 1

        self.options = {'ents': list(colors.keys()), 'colors': colors}

    def serve(self):
        displacy.serve(self.annotations, style='ent', manual=True, options=self.options)

    def render(self):
        html = displacy.render(self.annotations, style='ent', manual=True, options=self.options)
        with open('index.html', 'w') as f:
            f.write(html)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--text', type=str, help='raw text file to be annotated')
    parser.add_argument('--annotations', type=str, help='metamap annotations')
    args = parser.parse_args()

    print("Running...")

    parser = MetaMapAnnotator()

    with open(args.text, "r") as f:
        raw_text = f.read()

    with open(args.annotations, "r") as f:
        annotations = f.readlines()

    parser.annotate(raw_text, annotations)
    parser.define_colors(['#9b38bd', '#34c3b9', '#abd8d8', '#c2d54a', '#e0eaa9'])
    parser.serve()


