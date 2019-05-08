# Machine Learning - Gender Classification

Blog Gender Classification using Machine Learning.

For technical details, please refer to our [[paper]](https://github.com/michaeltran/ML-Gender-Classification/blob/master/docs/report.pdf).

## Requirements

```Rich Header Text
Python 3.6
Anaconda 5.2+
scikit-learn
TensorFlow
cuDNN
HDF5/h5py
Keras
Flair
textstat
pyspellchecker
```

---

## Problem Statement

> Given a set of labeled blogs written by males and females, predict the gender of the author of a new blog.

## Dataset

> Dataset 1: Sample blog author dataset used in [Mukherjee and Liu, EMNLP 2010] available from: [http://www.cs.uic.edu/~liub/FBS/blog-gender-dataset.rar](http://www.cs.uic.edu/~liub/FBS/blog-gender-dataset.rar).

> Dataset 2: Blog Authorship Corpus - consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person. Available from: [http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm).

---

## How to Run

### Data

Dataset 1 is already preloaded into this repository. Dataset 2 will need to be downloaded seperately from the [website](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) and placed into the folder `GenderClassification/GenderClassification/data/blogs`. Note that we only use a subset of this dataset in our project (records 0005114-3407543) as we do not need that much extra data to work on.

### Preprocessing

From the `GenderClassification/GenderClassification` directory, run the following command:

```Rich Text Format
python DataSplitter.py
```

This script will preprocess all of the data from Dataset 1 and Dataset 2, and will save the data into `test_data.xlsx`, `train_data.xlsx`, and `unlabeled_data.xlsx`. It will also perform POS pattern mining and save the results into `POSPatterns.txt`.

A brief overview on what is preprocessed:

* Classification (-1, 1)
* Text (Punctuation Normalization)
* Tokenized Text (Lemmatization and Stopword Removal)
* Tokenized Text 2 (NLTK Tokenization)
* POS
* Tagged POS (Word_POS)
* Word Count
* (Text) Length
* F-Measure
* Gender Preferential Features
* Factor Analysis
* Lexicon Count
* Text Standard (Estimated school grade level required to understand the text)

### Classification

From the `GenderClassification/GenderClassification` directory, run the following command:

```Rich Text Format
python GenderClassification.py
```

This script will run all of the preprocessed data through the classifiers. A brief overview on what is done:

* Dataset 1 train/test set will be tested (the baseline).
* Dataset 1 train/test set will be merged back into the total dataset and then we add some of dataset 2 into it (3000 records) for our total dataset.
* A new 90/10 training/validation set will be generated from the total dataset.
* "Supervised Semi-Supervised" learning will be performed and the training dataset will be expanded on by Dataset 2.
* Various Classifiers (NB/SVM/MLP/Keras) will be run on this new training and validation dataset and our final results will be reported.
