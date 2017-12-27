#!/usr/bin/env python

from __future__ import division
import argparse
from os import getenv

import torch

import onmt
import onmt.io
import onmt.ModelConstructor
import opts

from six.moves import zip

from sanic import Sanic, response
from sanic.request import Request
from sanic.exceptions import abort, InvalidUsage, ServerError
from sanic.log import error_logger

import nltk

app = Sanic(__name__)


class OnlineTranslator:
    def __init__(self, translator):
        assert isinstance(translator, onmt.translate.Translator)
        self.translator = translator

    def translate(self, sentences, device=-1,
                  batch_size=32, n_best=3, min_score=0,
                  round_score=False, round_to=3,
                  tokenize=False, alpha=0.0, beta=-0.0,
                  beam_size=5):
        self.translator.n_best = n_best
        self.translator.beam_size = beam_size

        # Translator
        scorer = onmt.translate.GNMTGlobalScorer(alpha, beta)
        self.translator.global_scorer = scorer

        if tokenize:
            sentences = (' '.join(nltk.word_tokenize(sentence))
                         for sentence in sentences)

        data = onmt.io.build_dataset_live(self.translator.fields,
                                          sentences,
                                          use_filter_pred=False)

        test_data = onmt.io.OrderedIterator(
            dataset=data, device=device,
            batch_size=batch_size, train=False, sort=False,
            shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.translator.fields,
            n_best, replace_unk=True, has_tgt=False)

        sents_preds = []
        for batch in test_data:
            batch_data = self.translator.translate_batch(batch, data)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                source_sentence = ' '.join(trans.src_raw)

                sent_preds = []
                predictions = trans.pred_sents
                scores = trans.pred_scores
                for prediction, score in zip(predictions, scores):
                    if round_score:
                        score = round(score, round_to)
                    if score < min_score:
                        break

                    prediction = ' '.join(prediction)
                    sent_preds.append({'prediction': prediction,
                                       'score': score})
                sents_preds.append({'sentence': source_sentence,
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

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # We need this to successfully instantiate the translator
    # but the beam size will be redefined with each query to translate
    beam_size = 1
    translator = onmt.translate.Translator(model, fields, beam_size,
                                           max_length=opt.max_sent_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=opt.cuda)
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
    alpha = data.get('alpha', 0.0)
    beta = data.get('beta', 0.0)
    beam_size = data.get('beam_size', 5)

    results = online_translator.translate(documents, device=opt.gpu,
                                          batch_size=batch_size, n_best=n_best,
                                          min_score=min_score, round_score=round_score,
                                          tokenize=should_tokenize,
                                          alpha=alpha, beta=beta,
                                          beam_size=beam_size)
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

