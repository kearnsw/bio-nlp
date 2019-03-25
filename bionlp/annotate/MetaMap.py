import subprocess
import json
import re
from pymetamap import MetaMap


def named_entity_recognition(text):
    mm = MetaMap.get_instance('/Users/kearnsw/programs/public_mm/bin/metamap16')
    concepts, error = mm.extract_concepts([text])
    return concepts


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
