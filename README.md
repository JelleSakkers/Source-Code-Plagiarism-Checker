## Overview

![report-1](https://github.com/niconap/PSE/assets/117186617/b76f1e5f-9838-4789-a888-3ce6f5518c01)

The script used the concepts outlined in [this paper](https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf) to perform code similarity checks based on code fingerprints and containment calculations. The script uses Numpy for efficient indexing calculations, while code preprocessing leverages [Pygments](https://pygments.org/), supporting tokenization across numerous programming languages.

The main desired and realised properties of the model is the usage of whitespace insensitivity. In matching source code files, matches should be unaffected by such things as extra whitespace, capitalization, punctuation, etc. Also, noise suppression, discovering short matches, such as the fact that the word the appears in two different documents, is uninteresting. Position independence Coarse-grained permutation of the contents of a document (e.g., scrambling the order of functions) should not affect the set of discovered matches.

## Usage
To check code similarity between two files. The most simplest usage is of the form below for wich default.
```bash
python3 code_similarity_checker.py file1.ext file2.ext --lang <python or c>
```

The script also offers a more sophisticated by setting customizable options such as adjusting the window size and the length of k-grams. These parameters help tune the sensitivity of the model. Adjusting the window size 
is essentially adjusting the minimum amount of matching characters between two source code files for wich a sequence of characters can be detected as plagiarism. Also, the length of k-grams is used to changed the number of gurantee matches for wich the model could detect two sequences of characters as plagiarism. So adjusting the sensitiviy of the model is really about configuring the noise and the guarantee size of matches of the model. 

```bash
python3 code_similarity_checker.py file1.ext file2.ext --lang <python or c> --k <k-gram length> --win_size <window size>
```

## Key Components
The main components consist of:

* The preprocessor.
* The core of the algorithm: Winnowing.
* Building an HTML report in which highlighting can take place by means of a mapping that translates token locations to the original location of the code.

## Detailed Workflow
_Note_: This section describes how this algorithm is implemented in the server backend, not how to run the algorithm independently, which is explained in an earlier section.

When submitting a submission, it will first be handled as if it were not plagiarism. After a short timeout, a background thread is started that checks for plagiarism. First, the submitted code is linked to its challenge, after which all other submissions within this challenge are compared with the submissions already submitted. All these submissions first go through the preprocessor. The code is tokenized after which it is simplified. This simplification helps to make the algorithm insensitive to renaming variables and functions. After the preprocessor, the Winnowing algorithm itself is executed. The linked paper describes in detail how this works. After running this algorithm, the plagiarism score is checked; this score is always between 0 and 1. A score of 0 is absolutely no match, 1 is a complete match. If the score is higher than the set threshold, the next steps will be carried out. If all submissions in the database are not a match, then of course nothing will happen. If there is a match, the submission is put into the HTML report generator. Using the token mapping - as generated in the preprocessor phase - this will be able to highlight the original submitted code in the places where the algorithm was triggered. Both the destination of stolen code and the source of stolen code are shown in the report. Once this report has been generated, the report will be sent to the KodeGrate mail.

## Note
The marking of plagiarized code is implemented at the token level (e.g., characterized by Pygments). This means that during the preprocessing, the start and end locations are tracked in both the original code and the preprocessed code. When detection occurs, we refer back to these start and end locations and a complete token is marked. The rationale behind this was that many elements are transformed into their respective single characters during preprocessing. This means that we actually mark the preceding unprocessed characters. Consequently, this code may not function properly with normal texts.
