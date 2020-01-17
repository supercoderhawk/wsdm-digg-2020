# WSDM 2020 Workshop
https://biendata.com/competition/wsdm2020/

ID: @nlp-rabbit 

## Prerequirements
Python >= 3.6

## Reproduce the result 
### Clone Code and Install Requirements

```bash
git clone https://github.com/supercoderhawk/wsdm-digg-2020
pip3 install -r requirements.txt
python3 -m spacy download en
```

### Setup ElasticSearch
1. setup elasticsearch service, refer to [link](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html)

2. setting value `ES_BASE_URL` in constants.py with your  configured elastic search endpoint.

### Prepare Data
1. unzip file and put all files under `data/` folder, **rename `test.csv` to `test_release.csv`**

2. Download [model](https://www.dropbox.com/s/6zcydsyf8tcgs7l/submit_model.zip?dl=0) , unzip it and put files into `data`
 folder

3. execute `bash scripts/prepare_data.sh` in **project root folder** to build the data for next step

### Execute the retrieval process end2end

* execute `bash scripts/run_end2end.sh` in **project root folder**

### Details
the above script includes three main parts

1. execute elasticsearch to retrieval candidate papers

    core logic in `search\search.py` which is called by `benchmark\benchmark.py`
    
2. execute the rerank by BERT

    core logic in `reranking\predict.py`, model code in `reranking\plm_rerank.py`

## Basic Algorithm Architecture  

1. recall phase
    1. keywords and keyphrase extraction
        1. noun chunk extraction 

        2. textrank keyword extraction

        3. candidate keywords filtering, including noun, proper noun and adjective

    2. BM25 based search (elasticsearch) 

2. rerank phase
    
    Bert based rerank (SciBert from AllenAI), single model, not have any ensemble methods
    
    training data built by first stage (BM25) search result
    loss is marginal loss (hinge loss)

## Train the Model

The model required to be trained just the Bert based reranking model

```bash
# prepare training data for reranking
bash scripts/prepare_rerank.sh

# training the rerank model
bash scripts/train_rerank.sh

# predict the result
bash scripts/predict_rerank.sh

```

## Others
1. In this project, abbreviation `plm` means `Pretrained Language Model`.

2. methods tried but **not effective**:
    1. Bert-Knrm, Bert-ConvKnrm paper: [CEDR: Contextualized Embeddings for Document Ranking](https://arxiv.org/abs/1904.07094), code in `reranking\plm_knrm.py` and `reranking\plm_conv_knrm.py`

    2. Bert based sentence vectorization method, paper [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175) (Use BERT CLS output replaced vanilla transformer trained from scratch) code in `vectorization\plm_vectorization.py` and `vectorization\predict.py`
    
    
## related papaer

[1] [Understanding the Behaviors of BERT in Ranking
](https://arxiv.org/abs/1904.07531)
    