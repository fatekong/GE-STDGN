# GE-STDGN
 Spatio-temporal weather prediction model based on graph evolution
===
## Requirement
* pytorch (see https://pytorch.org/get-started/locally/ for how to install it)
* GPU
* Numpy (see http://www.numpy.org/)

## Dataset

The source dataset is in [Google Drive](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view) and [Baiduyun](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view) with code `ni44`.

The source dataset and processed dataset are in [Google Drive](https://drive.google.com/drive/folders/1dWsPYqnkNcZi4s4WDTDAnOI359Lot2YE?usp=sharing)


## Directory description

├── Readme.md                   // help
├── requirements.txt            // 
├── GE                          // graph evolution
│   ├── Geo_threshold.py        // the adjacency matrix generation method based on geographical threshold
│   ├── KNN.py                  // the adjacency matrix generation method based on k nearest-neighbor 
│   ├── TIN.py                  // the adjacency matrix generation method based on triangulated irregular network
│   ├── parallel.py             // multi-GPUs parapall control
│   ├── population.py           // define about population
│   ├── utils.py                // define about genetic operations
│   └── individual.py           // define about individuals
├── baseline.py                 // baselines based on traditional machine learning methods
├── feature_selection.py        // feature selection
├── graph_evolution.py          // setup of graph evolution
├── main.py
├── measure.py                  // evaluation standard
├── model.py                    // model of STDGN
├── model_fs.py                 // model used in feature selection
├── tester.py
└── trainer.py

## Expiremental setup

### graph generation
* k->[6,23], \beta between distance and altitude is 0.8 in KNN
* threshold in distance is 300km, and altitude is 1200m

### graph evolution 
* mutation 0.1
* crossing 0.8


 
