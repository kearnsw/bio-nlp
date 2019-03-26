import subprocess
import json
import re
from pymetamap import MetaMap


class UMLSConcept:
    def __init__(self, concept):
        self.semtypes = concept.semtypes.replace("[", "").replace("]", "").split(",")
        self.cui = concept.cui
        self.score = concept.score
        self.preferred_name = concept.preferred_name
        self.trigger = concept.trigger


def named_entity_recognition(text):
    mm = MetaMap.get_instance('/Users/kearnsw/programs/public_mm/bin/metamap16')
    concepts, error = mm.extract_concepts([text])

    return [UMLSConcept(concept) for concept in concepts]


def run_metamap(text, _format=None):
    pipe = subprocess.Popen(['echo', str(text)], stdout=subprocess.PIPE)
    metamap = subprocess.Popen(['metamap', '-I', '--JSONn', '--silent'], stdin=pipe.stdout, stdout=subprocess.PIPE)
    output = metamap.communicate()[0]

    if _format == "json":
        output = output.split(b"\n")
        print(output)
        return json.loads(output[1])
    else:
        return output


def get_lexical_form(concept):
    matches = re.findall(r'"\w+.*?"', concept.trigger)
    print(concept.trigger)
    if len(matches) >= 2:
        return matches[1].replace('"', '')
    elif len(matches) == 1:
        return matches[0].replace('"', '')
    else:
        return None


if __name__ == "__main__":
    white_list = ["antb", "neop", "dsyn", "vita", "virs", "phsu", "phsf", "clnd", "bpoc", "anab", "cell", "bacs"]
    concepts = named_entity_recognition("cholesterol")
    for concept in concepts:
        print(type(concept.semtypes))
        print([True if semtype in white_list else False for semtype in concept.semtypes])
        if any([semtype in white_list for semtype in concept.semtypes]):
            pass
        else:
            print(concept.semtypes)
