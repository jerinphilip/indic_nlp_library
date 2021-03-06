# Indic NLP Library

The goal of the Indic NLP Library is to build Python based libraries for common text processing and Natural Language Processing in Indian languages. Indian languages share a lot of similarity in terms of script, phonology, language syntax, etc. and this library is an attempt to provide a general solution to very commonly required toolsets for Indian language text.

The library provides the following functionalities:

- Text Normalization
- Script Information
- Tokenization
- Word Segmentation
- Script Conversion
- Romanization
- Indicization
- Transliteration
- Translation

The data resources required by the Indic NLP Library are hosted in a different repository. These resources are required for some modules. You can download from the [Indic NLP Resources](https://github.com/anoopkunchukuttan/indic_nlp_resources) project.

## Pre-requisites

- Python 3.x (For Python 2.x version check the tag `PYTHON_2.7_FINAL_JAN_2019`. Not actively supporting Python 2.x anymore, but will try to maintain as much compatibility as possible)
- [Morfessor 2.0 Python Library](http://www.cis.hut.fi/projects/morpho/morfessor2.shtml)
- [Indic NLP Resources](https://github.com/anoopkunchukuttan/indic_nlp_resources)

## Configuration

The following installs the module `indicnlp` into your python packages.

```
$ python3 setup.py install \
    --user # optional
```

For development, you're advised to use a `virtualenv.


## Usage 

- Python API: Check [this IPython Notebook](http://nbviewer.ipython.org/url/anoopkunchukuttan.github.io/indic_nlp_library/doc/indic_nlp_examples.ipynb) for examples.
  - You can find the Python 2.x Notebook [here](http://nbviewer.ipython.org/url/anoopkunchukuttan.github.io/indic_nlp_library/doc/indic_nlp_examples_2_7.ipynb) 
- Commandline Interface: The commandline interface is documented on the [project website](http://anoopkunchukuttan.github.io/indic_nlp_library)

## Website

`http://anoopkunchukuttan.github.io/indic_nlp_library`

## Author
Anoop Kunchukuttan ( anoop.kunchukuttan@gmail.com )

## Version: 0.4

## Revision Log

0.4 : 28 Jan 2019: Ported to Python 3, and lots of feature additions since last release; primarily around script information, script similarity and syllabification.

0.3 : 21 Oct 2014: Supports morph-analysis between Indian languages

0.2 : 13 Jun 2014: Supports transliteration between Indian languages and tokenization of Indian languages 

0.1 : 12 Mar 2014: Initial version. Supports text normalization.

## LICENSE

Copyright Anoop Kunchukuttan 2013 - present
 
Indic NLP Library is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Indic NLP Library is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License 
along with Indic NLP Library.  If not, see <http://www.gnu.org/licenses/>.

