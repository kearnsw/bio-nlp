import subprocess
import json


def run_metamap(text):
    pipe = subprocess.Popen(['echo', str(text)], stdout=subprocess.PIPE)
    metamap = subprocess.Popen(['metamap', '-I', '--JSONn', '--silent'], stdin=pipe.stdout, stdout=subprocess.PIPE)
    output = metamap.communicate()[0]
    output = output.split(b"\n")
    return json.loads(output[1])

if __name__ == "__main__":
    res = run_metamap("advil tylenol")
    print(res)