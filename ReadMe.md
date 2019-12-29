# WSDM 2020 Workshop

## Prerequirements
Python >= 3.6

## Usage
### Install Requirements

```bash
pip3 install -r requirements.txt
python3 -m spacy download en

```

### Setup ElasticSearch
1. setup elasticsearch service, refer to [link](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html)

2. setting value `ES_BASE_URL` in constants.py with your  configured elastic search endpoint.

### Prepare Data
1. unzip data files to `data/` folder
2. execute `scripts/prepare_data.sh`

### Execute the code
1. elasticsearch

2. prepare rerank data from elastic search result

3. 