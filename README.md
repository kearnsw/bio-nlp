# NLP Challenges for Detecting Medication and Adverse Drug Events from Electronic Health Records (MADE1.0)
_hosted by University of Massachusetts Medical School_

Adverse drug events (ADEs) are common and occur in approximately 2-5% of hospitalized adult patients. Each ADE is estimated to increase healthcare cost by more than $3,200. Severe ADEs rank among the top 5 or 6 leading causes of death in the United States. Prevention, early detection and mitigation of ADEs could save both lives and dollars. Employing natural language processing (NLP) techniques on electronic health records (EHRs) provides an effective way of real-time pharmacovigilance and drug safety surveillance.

We’ve annotated 1092 EHR notes with medications, as well as relations to their corresponding attributes, indications and adverse events. It provides valuable resources to develop NLP systems to automatically detect those clinically important entities. Therefore we are happy to announce a public NLP challenge, MADE1.0, aiming to promote deep innovations in related research tasks, and bring researchers and professionals together exchanging research ideas and sharing expertise. The ultimate goal is to further advance ADE detection techniques to improve patient safety and health care quality.

## Tentative Timelines

    Registration: begins August 1st, 2017
    Training data release: November 1st, 2017
    System submission: Feb 1st, 2018
    Workshop: in conjunction with AMIA summit 2018, March 2018

## Annotated Data

The entire dataset contains 1092 de-identified EHR notes from 21 cancer patients. Each EHR note was annotated with medication information (medication name, dosage, route, frequency, duration), ADEs, indications, other signs and symptoms, and relations among those entities. We split the data into a training set consisting of ~900 notes and a test set consisting of ~180 notes. Both will be released in BioC format.

Participanting groups must sign a Data Recepient Agreement and provide their own institute IRB approval. Registered users can download the Data Recepient Agreement here. Each participating team only needs to submit one Data Recepient Agreement and one IRB approval notice. All team members listed on the IRB are allowed to use the data. Click here to upload completed Date Recepient Agreement. 
Task Definitions

## MADE1.0 challenge consists of three tasks defined as follows.

1. Named entity recognition (NER): develop systems to automatically detect mentions of medication name and its attributes (dosage, frequency, route, duration), as well as mentions of ADEs, indications, other signs & symptoms.

2. Relation identification (RI): given the truth entity annotations, build system to identify relations between medication name entities and its attribute entities, as well as relations between medication name entities and ADE, indications and other sign & symptoms.

3. Integrated task (IT): design and develop a integrative system to conduct the above two tasks together.

## Evaluations

All the participated teams will not get the test data itself, instead we will have a platform available to accept system submissions where all the submitted systems will be run on the same withheld test data, in the same computing environment and evaluated by the same evaluation scripts with the same metrics on different tasks. Each team is allowed to submit up to three versions of systems for each task, and each team can choose to participate either one or more tasks in this challenge.

Below are the publications of two baseline systems. In this competition, we used a different approach to partition the training and testing data. Therefore, the performance of the two baseline systems can be used as an approximation only.

1. Structured prediction models for RNN based sequence labeling in clinical text. (Abhyuday Jagannatha, Hong Yu; EMNLP 2016)

2. Bidirectional Recurrent Neural Networks for Medical Event Detection in Electronic Health Records. (Abhyuday Jagannatha, Hong Yu; NAACL HLT 2016)
Dissemination

Participants are asked to submit a short paper describing their methodologies. It can contain a graphical summary of the proposed architecture. The document should not exceed 4 pages, 1.5 line spacing, 12 font size. The authors of either top performing systems or particularly novel approaches will be invited to present or demonstrate their systems at the workshop. A special issue of a journal will be organized following the workshop.

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
