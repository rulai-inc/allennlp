import logging
import time
import uvicorn
import pdb
from fastapi import FastAPI
import logging
import torch

from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, WhitespaceTokenizer

from config import load_config, setup_logging
#from src import run_text, run_session

logger = logging.getLogger(__name__)
conf = load_config()
app = FastAPI()

if torch.cuda.is_available():
    srl_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz", cuda_device=torch.cuda.current_device())
else:
    srl_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

# A hack for making the labeller use white space tokenizer
# Assume the text is already tokenized
srl_predictor._tokenizer.spacy.tokenizer = WhitespaceTokenizer(srl_predictor._tokenizer.spacy.vocab)
sent_tokenizer = SpacySentenceSplitter()

batch_size = conf['batch_size']
empty_text_res = {
    'verbs': [],
    'words': ''
}


def gen_res(doc):
    verbs = doc['verbs']
    verbs_json = [{'verb': verb['verb'], 'tags': verb['tags']} for verb in verbs]
    doc['verbs'] = verbs_json
    return doc


@app.post('/srl_parse')
async def srl_parse(data: dict):
    text = data['text']
    sents = sent_tokenizer.split_sentences(text)
    jsons = [{"sentence": sent} for sent in sents]
    docs = srl_predictor.predict_batch_json(jsons)
    res = [gen_res(doc) for doc in docs]

    return res


@app.post('/srl_parse_session')
async def srl_parse_session(data: dict):
    text_list = data['session']
    sent_list = []
    lengths = []
    if not data['split']:
        for sents in text_list:
            sent_list.extend(sents)
            lengths.append(len(sents))
    else:
        for text in text_list:
            splits = sent_tokenizer.split_sentences(text)
            sent_list.extend(splits)
            lengths.append(len(splits))

    # remove empty texts and record their indices
    full_number = len(sent_list)
    orig_sent_list = sent_list.copy()
    empty_ind = {i: True for i in range(len(sent_list)) if sent_list[i] == ''}
    sent_list = [sent for sent in sent_list if sent]

    res_list = []
    start = 0
    while start < len(sent_list):
        end = min(start + batch_size, len(sent_list))
        jsons = [{"sentence": sent} for sent in sent_list[start:end]]
        docs = srl_predictor.predict_batch_json(jsons)
        res = [gen_res(doc) for doc in docs]
        res_list.extend(res)
        start += batch_size

    # insert empty results for empty text
    full_res_list = [empty_text_res for x in range(full_number)]
    res_list_ind = 0
    empty_count = 0
    for i in range(full_number):
        if i not in empty_ind:
            full_res_list[i] = res_list[res_list_ind]
            res_list_ind += 1
        else:
            empty_count += 1

    assert res_list_ind == len(res_list) and empty_count == len(empty_ind)

    response = []
    start = 0
    end = start
    for length in lengths:
        end = start + length
        response.append(full_res_list[start:end])
        start = end

    assert end == full_number

    return response


if __name__ == "__main__":
    #setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=conf["server_port"])
