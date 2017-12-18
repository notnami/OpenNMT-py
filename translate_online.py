#!/usr/bin/env python

from __future__ import division
import argparse
from os import getenv
import torch

import onmt
import onmt.IO
import opts
import translate

from pprint import pprint

from six.moves import zip_longest
from six.moves import zip

from sanic import Sanic, response
from sanic.request import Request
from sanic.exceptions import abort, InvalidUsage, ServerError
from sanic.log import error_logger

import nltk

app = Sanic(__name__)


class OnlineTranslator:
    def __init__(self, translator):
        assert isinstance(translator, onmt.Translator)
        self.translator = translator

    def translate(self, sentences, device=-1,
                  batch_size=32, n_best=3, min_score=0,
                  round_score=False, round_to=3,
                  tokenize=False):

        if tokenize:
            sentences = (' '.join(nltk.word_tokenize(sentence))
                         for sentence in sentences)

        self.translator.opt.n_best = n_best

        data = onmt.IO.build_dataset_live(self.translator.fields,
                                          sentences,
                                          use_filter_pred=False)

        test_data = onmt.IO.OrderedIterator(dataset=data, device=device,
                                            batch_size=batch_size, train=False,
                                            sort=False, shuffle=False)

        pred_score_total, pred_words_total = 0, 0
        sents_preds = []
        for batch in test_data:
            pred_batch, gold_batch, pred_scores, gold_scores, attn, src, indices \
                = self.translator.translate(batch, data)
            pred_score_total += sum(score[0] for score in pred_scores)
            pred_words_total += sum(len(x[0]) for x in pred_batch)

            # z_batch: an iterator over the predictions, their scores,
            # the gold sentence, its score, and the source sentence for each
            # sentence in the batch. It has to be zip_longest instead of
            # plain-old zip because the gold_batch has length 0 if the target
            # is not included.
            z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))

            for (pred_sents, gold_sent,
                 pred_score, gold_score, src_sent) in z_batch:
                n_best_preds = [" ".join(pred) for pred in pred_sents[:n_best]]
                sent_preds = []
                for pred, score in zip(n_best_preds, pred_score):
                    # if a minimum score has been set,
                    # and we have already given at least one prediction
                    # stop here
                    if round_score:
                        score = round(score, round_to)
                    if min_score and sent_preds and score < min_score:
                        break
                    sent_preds.append({'prediction': pred,
                                       'score': score})
                words = translate.get_src_words(
                    src_sent, self.translator.fields["src"].vocab.itos)
                sents_preds.append({'sentence': words,
                                    'predictions': sent_preds})
        return sents_preds


def get_translator():
    # NOTE: I don't like that we have to do this "dummy_parser"
    # but this is how OpenNMT's Translator objects are instantiated
    # (Anton)
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    online_translator = OnlineTranslator(translator)
    return online_translator


@app.route('/')
async def home(request: Request):
    return response.json(
        {'message': 'The OpenNMT model is live.'})


@app.route('/predict/', methods=['POST'])
async def predict(request: Request):
    data = request.json
    if not data:
        abort(400, 'Got request without JSON data.')
    documents = data['documents']
    batch_size = data.get('batch_size', 32)
    n_best = data.get('n_best', 3)
    min_score = data.get('min_score', -2.0)
    round_score = data.get('round_score', False)
    should_tokenize = data.get('tokenize', True)

    results = online_translator.translate(documents, device=opt.gpu,
                                          batch_size=batch_size, n_best=n_best,
                                          min_score=min_score, round_score=round_score,
                                          tokenize=should_tokenize)
    return response.json({'predictions': results})


@app.exception([InvalidUsage, ServerError])
def handle_app_errors(request, exception):
    error_code = exception.status_code
    message = (f'Got {exception} ({error_code}) '
               f'processing the following request:\n{request.body}')
    error_logger.error(message)
    return response.json({'status': 'Error',
                          'message': "The server couldn't process your request."},
                         status=error_code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='translate_online.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    online_translator = get_translator()

    host = getenv('NMT_HOST', 'localhost')
    port = getenv('NMT_PORT', '9999')
    app.run(host=host, port=port)
