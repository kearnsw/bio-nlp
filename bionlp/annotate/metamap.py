from bionlp.core.LinkedList import LinkedList
from argparse import ArgumentParser
import sys

class MetaMapAnnotator(object):
	
	def __init__(self):
		pass

	def annotate(self, text, annotations):
		
		# Convert text to char linked list
		ll = LinkedList()
		ll.from_string(text)
		
		for ann in annotations:
			ann = ann.split("|")
			if len(ann) >= 8:
				term = ann[5]
				offset = ann[7]
				print(term, offset)	
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

