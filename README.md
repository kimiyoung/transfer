# Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks

## Intro

This is an implementation of the paper
```
Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks
Zhilin Yang, Ruslan Salakhutdino, William W. Cohen
ICLR 2017
```

## Requirements

Lasagne, Theano. Python 2.7.

Install Lasagne and Theano with the instructions here: https://github.com/Lasagne/Lasagne#installation

## Get data

### Publicly Available Data

Some of the datasets are publicly available, which can be downloaded from our server.
```
wget http://kimi.ml.cmu.edu/transfer/data.tar.gz
tar -xvzf data.tar.gz
```
The above command will download the Genia and Twitter datasets, along with the Senna embeddings and an English gazeteer.

Other datasets require a LDC license; please contact your institution to access the below datasets.

### Get Chunking Dataset

Get the CoNLL 2000 chunking dataset using a LDC license, and organize the files with the following structure:
```
transfer/chunking/train.txt
transfer/chunking/test.txt
```

### Get POS Dataset

Get the PennTreebank 2003 dataset using a LDC license, and organize the files with the following structure:
```
transfer/pos_tree/dev.txt
transfer/pos_tree/test.txt
transfer/pos_tree/train.txt
```

### Get Spanish NER Dataset

Get the CoNLL 2003 Spanish NER dataset using a LDC license, and organize the files with the following structure:
```
transfer/span/esp.testa
transfer/span/esp.testb
transfer/span/esp.train
```

### Get English NER dataset

Get the CoNLL 2003 English NER dataset using a LDC license, and organize the files with the following structure:
```
transfer/eng.testa.old
transfer/eng.testb.old
transfer/eng.train
```

## Labeling Rates and Data Splits

For each dataset, we first concatenate the training set and the dev set (training set always first). And then use the following function (in `sample.py`) to sample a list of indices that are used for training.
```
def create_sample_index(rate, len):
    np.random.seed(13)
    return np.random.choice(len, int(rate * len))
```
where `rate` is the labeling rate, and `len` is the number of instances (training+dev). The function will return an np array of indices; other instances not in the list will be discarded during training.

You can use the above function to reproduce the data splits for comparison of different models.

## Transfer Learning with Our Model

The transfer learning scripts are in `joint.py` and `lang.joint.py`, where `joint.py` is used for transfer learning within one language, and `lang.joint.py` is used to cross-lingual transfer learning.

`joint.py` accepts the following input formats:
```
python2 joint.py --tasks <target_task_name> <source_task_name> --labeling_rates <labeling_rate_for_target_task> <labeling_rate_for_source_task> [--very_top_joint]
```
where task names come from the list
```
[genia, pos, ner, chunking, ner_span, twitter_ner, twitter_pos]
```
and labeling rates are float numbers. The flag `very_top_joint` indicates whether to share the parameters of the CRF layer or not.

Below are examples of the transfer learning settings used in our paper (Fig. 2):
```
# transfer from PTB to Genia
python2 joint.py --tasks genia pos --labeling_rates <labeling_rate> 1.0 --very_top_joint

# transfer from CoNLL 2003 NER to Genia
python2 joint.py --tasks genia ner --labeling_rates <labeling_rate> 1.0

# transfer from Spanish NER to Genia
python2 lang.joint.py --tasks genia ner_span --labeling_rates <labeling_rate> 1.0

# transfer from PTB to Twitter POS tagging
python2 joint.py --tasks twitter_pos pos --labeling_rates <labeling_rate> 1.0

# transfer from CoNLL 2003 to Twitter NER
python2 joint.py --tasks twitter_ner ner --labeling_rates <labeling_rate> 1.0

# transfer from CoNLL 2003 NER to PTB POS tagging
python2 joint.py --tasks pos ner --labeling_rates <labeling_rate> 1.0

# transfer from PTB POS tagging to CoNLL 2000 chunking
python2 joint.py --tasks chunking pos --labeling_rates <labeling_rate> 1.0

# transfer from PTB POS tagging to CoNLL 2003 NER
python2 joint.py --tasks ner pos --labeling_rates <labeling_rate> 1.0

# transfer from CoNLL 2003 English NER to Spanish NER
python2 lang.joint.py --tasks ner_span ner --labeling_rates <labeling_rate> 1.0

# transfer from Spanish NER to CoNLL 2003 English NER
python2 lang.joint.py --tasks ner ner_span --labeling_rates <labeling_rate> 1.0
```

## Our Results (With More Results Than in the Paper)

Target | Source | Labeling Rate | With Transfer | Without Transfer
--------|--------|-------:|--------|-------- 
genia | PTB | 0.0 | 0.840899499608 | N/A
genia | PTB | 0.001 | 0.916581258415 | 0.832640019292
genia | PTB | 0.01 | 0.963083539318 | 0.935592130383
genia | PTB | 0.1 | 0.981953738872 | 0.978035007335
genia | PTB | 1.0 | 0.990092642833 | 0.990655332489
genia | Eng NER | 0.001 | 0.87471269687 | 0.832640019292
genia | Eng NER | 0.01 | 0.941942485079 | 0.935592130383
genia | Eng NER | 0.1 | 0.979944132956 | 0.978035007335
genia | Eng NER | 1.0 | 0.989951970419 | 0.990655332489
genia | Span NER | 0.001 | 0.843853620305 | 0.832640019292
genia | Span NER | 0.01 | 0.93111070919 | 0.935592130383
genia | Span NER | 0.1 | 0.978718273347 | 0.978035007335
genia | Span NER | 1.0 | 0.989550049235 | 0.990655332489
PTB | Eng NER | 0.001 | 0.87471269687 | 0.841578354698
PTB | Eng NER | 0.01 | 0.949326669443 | 0.942871025961
PTB | Eng NER | 0.1 | 0.967891464976 | 0.965916979037
PTB | Eng NER | 1.0 | 0.974470513829 | 0.975334351428
Eng NER | PTB | 0.001 | 0.346473029046 | 0.335092085615
Eng NER | PTB | 0.01 | 0.749249658936 | 0.686385971674
Eng NER | PTB | 0.1 | 0.870218090812 | 0.86219588832
Eng NER | PTB | 1.0 | 0.91264717787 | 0.91208817241
Chunking | PTB | 0.001 | 0.622235477654 | 0.58375524895
Chunking | PTB | 0.01 | 0.867262565155 | 0.834900974403
Chunking | PTB | 0.1 | 0.927242176013 | 0.90649356106
Chunking | PTB | 1.0 | 0.953936031606 | 0.945709723506
Eng NER | Span NER | 0.001 | 0.346253229974 | 0.335092085615
Eng NER | Span NER | 0.01 | 0.726148735929 | 0.686385971674
Eng NER | Span NER | 0.1 | 0.865126276196 | 0.86219588832
Eng NER | Span NER | 1.0 | 0.912161558395 | 0.91208817241
Span NER | Eng NER | 0.001 | 0.164485165794 | 0.115025161754
Span NER | Eng NER | 0.01 | 0.604273247066 | 0.598373003917
Span NER | Eng NER | 0.1 | 0.765227337718 | 0.745397008055
Span NER | Eng NER | 1.0 | 0.848126232742 | 0.846034214619
Twitter POS | PTB | 0.001 | 0.020282728949 | 0.00860479409957
Twitter POS | PTB | 0.01 | 0.646588813768 | 0.503380454825
Twitter POS | PTB | 0.1 | 0.836508912108 | 0.748002458513
Twitter POS | PTB | 1.0 | 0.907191149355 | 0.893054701905
Twitter NER | Eng NER | 0.001 | 0.0137931034483 | 0.00950118764846
Twitter NER | Eng NER | 0.01 | 0.24154589372 | 0.0963855421687
Twitter NER | Eng NER | 0.1 | 0.432432432432 | 0.346534653465
Twitter NER | Eng NER | 1.0 | 0.6473029045 | 0.63829787234

