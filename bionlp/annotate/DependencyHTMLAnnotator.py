from spacy import displacy

data = {
    'words': [
        {'text': 'Smoking', 'tag': 'NN'},
        {'text': 'causes', 'tag': 'VBZ'},
        {'text': 'lung', 'tag': 'ADJ'},
        {'text': 'cancer', 'tag': 'NN'},
        {'text': '.', 'tag': 'PUNC'},
        {'text': 'It', 'tag': 'PRP'},
        {'text': 'can', 'tag': 'AUX'},
        {'text': 'lead', 'tag': 'VBZ'},
        {'text': 'to', 'tag': 'IN'},
        {'text': 'death', 'tag': 'NN'},
        {'text': '.', 'tag': 'PUNC'}],
    'arcs': [
        {'start': 0, 'end': 5, 'label': 'Coref', 'dir': 'left'}
        ]
        }

html = displacy.serve(data, style='dep', manual=True)