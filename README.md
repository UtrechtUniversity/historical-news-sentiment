# Interest

This repository contains code for calculating the sentiment of historical news articles. While it has been tested on the [Delpher Kranten](https://www.delpher.nl/nl/kranten) corpus, the code is adaptable to analyze the sentiment of any given text. Using the [dataQuest](https://github.com/UtrechtUniversity/dataQuest?tab=readme-ov-file) Python package, articles related to fossil fuels (gas, coal, and oil) were filtered to train and test the models in this repository. The goal is to examine the evolving sentiment toward the use of these fuels between 1960 and 1995. The ```interest```Python package offers a variety of sentiment analysis methods, optimized and tested for accuracy in processing text data.

The interest Python package includes the following methods:

- An unsupervised approach that calculates the sentiment of articles by measuring the Euclidean distance between    positive, negative, and article word vectors using document vectors.

## Getting Started
Clone this repository to your working station to obtain examples and python scripts:

```https://github.com/UtrechtUniversity/historical-news-sentiment.git```

## Prerequisites

```
- Python [>=3.8, <3.11]
- jupyterlab (or any other program to run jupyter notebooks)
```
To install jupyterlab:

``` pip install jupyterlab```

## Installation

### Option 1 - Install Interest package
To run the project, ensure to install the interest package.

``` pip install interest ```

### Option 2 - Run from source code
If you want to run the scripts without installation you need to:

- Install requirement

```
pip install setuptools wheel
python -m pip install build
```
Change your current working directory to the location of your pyproject.toml file.

```
python -m build
pip install .
```

- Set PYTHONPATH environment: On Linux and Mac OS, you might have to set the PYTHONPATH environment variable to point to this directory.
```
export PYTHONPATH="current working directory/dataQuest:${PYTHONPATH}"
```

## Built with
- NumPy
- Pandas
- Gensim
- NLTK
- scikit-learn

## About the Project

### Date: May 2024

### Researcher(s):

- Pim Huijnen (p.huijnen@uu.nl)

### Research Software Engineer(s):

- Shiva Nadi (s.nadi@uu.nl)  
- Parisa Zahedi (p.zahedi@uu.nl)

## License

The code in this project is released under [MIT license](https://github.com/UtrechtUniversity/patent-breakthrough/blob/main/LICENSE).

## Contributing

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

### To contribute:

- Fork the Project
- Create your Feature Branch (git checkout -b feature/AmazingFeature)
- Commit your Changes (git commit -m 'Add some AmazingFeature')
- Push to the Branch (git push origin feature/AmazingFeature)
- Open a Pull Request

## Contact

Pim Huijnen - p.huijnen@uu.nl

Project Link:  https://github.com/UtrechtUniversity/historical-news-sentiment.git
