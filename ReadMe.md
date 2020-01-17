# WSDM 2020 Workshop
https://biendata.com/competition/wsdm2020/

ID: @nlp-rabbit 

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

2. Download [SciBert](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar) , unzip it to data folder

3. execute `bash scripts/prepare_data.sh` in **project root folder** to build the data for next step

### Execute the retrieval process end2end

* put the model into `data/models/rerank_model.model`

* execute `bash scripts/run_end2end.sh` in **project root folder**

#### details
the above script includes three main parts

1. execute elasticsearch to retrieval candidate papers

    core logic in `search\search.py` which is called by `benchmark\benchmark.py`

2. prepare rerank data from elastic search result (baseline result)

    core logic in `reranking\predict.py`, model in `reranking\plm_rerank.py`
3. execute the rerank by BERT

### others
1. In this project, abbreviation `plm` means `Pretrained Language Model`.

2. methods tried but not effective:
    1. Bert-Knrm, Bert-ConvKnrm paper: [CEDR: Contextualized Embeddings for Document Ranking](https://arxiv.org/abs/1904.07094), code in `reranking\plm_knrm.py` and `reranking\plm_conv_knrm.py`

    2. Bert based sentence vectorization method, paper [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175) (Use BERT CLS output replaced vanilla transformer trained from scratch) code in `vectorization\plm_vectorization.py` and `vectorization\predict.py`

## basic architecture  

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

## train the model

The model required to be trained just the Bert based reranking model

```bash
# prepare training data for reranking
bash scripts/prepare_rerank.sh

# training the rerank model
bash scripts/train_rerank.sh

# predict the result
bash scripts/predict_rerank.sh

```
