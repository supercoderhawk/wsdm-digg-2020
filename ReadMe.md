# WSDM 2020 Workshop

## Prerequirements
Python >= 3.6

## Reproduce the result
### Install Requirements

```bash
pip3 install -r requirements.txt
python3 -m spacy download en

```

### Setup ElasticSearch
1. setup elasticsearch service, refer to [link](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html)

2. setting value `ES_BASE_URL` in constants.py with your  configured elastic search endpoint.

### Prepare Data
1. unzip file and put all files under `data/` folder, rename `test.csv` to `test_release.csv`
2. execute `bash scripts/prepare_data.sh` in **project root folder** to build the data for next step

### Execute the retrieval process

execute `bash scripts/run_retrieval.sh` in **project root folder**

#### details
the above script includes three main parts

1. execute elasticsearch to retrieval candidate papers

2. prepare rerank data from elastic search result (baseline result)

3. execute the rerank by BERT

### others


## basic architecture  

1. recall phase

    noun chunk extraction + textrank keyword extraction + BM25 based search (elasticsearch) 

2. rerank phase
    
    Bert based rerank (SciBert from AllenAI)
    
## train the model

The model required to be trained in this project including the