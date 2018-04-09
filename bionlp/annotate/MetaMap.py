import subprocess
import json


def run_metamap(text, _format=None):
    pipe = subprocess.Popen(['echo', str(text)], stdout=subprocess.PIPE)
    metamap = subprocess.Popen(['metamap', '-I', '--JSONn', '--silent'], stdin=pipe.stdout, stdout=subprocess.PIPE)
    output = metamap.communicate()[0]
    if _format == "json":
        output = output.split(b"\n")
        return json.loads(output[1])
    else:
        return output
