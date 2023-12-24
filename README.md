## About this project
This is the source code for a research project that I did in collaboration with Dr. Hegler Tissot at Drexel University for my senior research project, which also happened to earn 1st place in the 2023 CCI Senior Research Project competition. 
The modules in this repository are tools that analyze vector embeddings and generate synthetic data based on those embeddings. The main focus is exploring the embeddings, and generating the synthetic data based on cluster sampling.

## How to install and run
1. clone the repo
2. set up a virtual Python env (PyCharm does this for you)
3. pip install -r requirements.txt
4. look in main.py
5. uncomment the lines of code that you're interested in in the main function
6. run main.py

There are some datasets in the datasets directory that you can play with, they come from https://archive.ics.uci.edu/

## Limitations
- This project does not work with multi-relational data (e.g. SQL database)
- The features and linkage matrix between embeddings is held in memory, which can be pretty large. Synthetic data generation may struggle or be very slow on machines with less RAM.
- Embeddings are included with the example datasets, but there is no way to train embeddings for a new dataset within this repo. You will have to supply them from somewhere else. That is not the focus of this repo.
- This synthetic generation method, if used with certain parameters, will expose individuals. It is not guaranteed to be compliant with Differential Privacy.
