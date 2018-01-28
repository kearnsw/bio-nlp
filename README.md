# Installing BioNLP
```
git clone https://github.com/kearnsw/bio-nlp.git
cd bio-nlp; python setup.py develop
```

## Install Pytorch
`conda install pytorch torchvision -c soumith`

## NER for Adverse Drug Reaction Detection
### Download Cadec data
Put the cadec data from `https://data.csiro.au/dap/landingpage?pid=csiro%3A10948` in the data directory.

### Download Word Vectors
Download word vectors into vectors folder
```
mkdir vectors; cd vectors; wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
You can download PubMed Word Embeddings from "https://drive.google.com/open?id=" with the following ids:
Window-size 2: 0B34vNkiF96SxaDZNUE5fOVdVWm8
Window-size 30: 0B34vNkiF96SxZ3JkWWxIbG9NVHM

## Running
If you followed the instructions above, you can use the run script. Otherwise you may need to supply your own arguments to the model. You can view these arguments with a description using the -h flag, e.g.
=======
### Running 
If you followed the instructions above, you can use the run script using the below command.
```shell
python -m bionlp.core.BiLSTM -e "vectors/glove.6B.100d.txt" -d 100 --text "data/cadec/text/" --ann "data/cadec/original/" --hidden 64 --epochs 100 --layers 2 --models "models"

```

Otherwise you may need to supply your own arguments to the model. You can view these arguments with a description using the -h flag, e.g.
```
python -m bionlp.core.BiLSTM -h
``` 

### NER Visualizer 

```
annotate file.txt
```

## Question Type Classification

### Download Genetic and Rare Disease data set
```
cd data; wget "https://ceb.nlm.nih.gov/ridem/infobot_docs/GARD_qdecomp.master.03.qtd.xml; cd .."
```

### Running Model
```
python -m bionlp.core.CNN_Classifier -e "vectors/glove.6B.100d.txt" -d 100  --text "data/GARD_qdecomp.master.03.qtd.xml" --epochs 100 --batch_size 10 --models "models/" --train --validate
```
