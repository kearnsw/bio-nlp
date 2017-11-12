# Installing BioNLP
```
git clone https://github.com/kearnsw/bio-nlp.git
cd bio-nlp; python setup.py develop
```

## Install Pytorch
`conda install pytorch torchvision -c soumith`

## Download Cadec data
Put the cadec data from `https://data.csiro.au/dap/landingpage?pid=csiro%3A10948` in the data directory.

## Download Word Vectors
Download word vectors into vectors folder
```
mkdir vectors; cd vectors; wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## Running 
If you followed the instructions above, you can use the run script. Otherwise you may need to supply your own arguments to the model. You can view these arguments with a description using the -h flag, e.g.
```
python bionlp/src/BiLSTM.py -h
``` 

## NER Visualizer 

```
annotate file.txt
```
