# -*- coding: utf-8 -*-

from onmt.io.IO import ONMTDatasetBase, _join_dicts, _peek, \
                       _construct_example_fromlist


class AudioDataset(ONMTDatasetBase):
    """ Dataset for data_type=='audio'

        Build Example objects, Field objects, and filter_pred function
        from audio corpus.

        Args:
            fields: a dictionary of Field objects.
            src_examples_iter: preprocessed source example_dict iterator.
            tgt_examples_iter: preprocessed target example_dict iterator.
            num_src_feats: number of source side features.
            num_tgt_feats: number of target side features.
            tgt_seq_length: maximum target sequence length.
            sample_rate: sample rate.
            window_size: window size for spectrogram in seconds.
            window_stride: window stride for spectrogram in seconds.
            window: indow type for spectrogram generation.
            normalize_audio: subtract spectrogram by mean and divide
                             by std or not.
            use_filter_pred: use a custom filter predicate to filter
                             examples?

    """
    def sort_key(self, ex):
        return -ex.src.size(1)

    def _process_corpus(self, fields, src_examples_iter, tgt_examples_iter,
                        num_src_feats=0, num_tgt_feats=0,
                        tgt_seq_length=0, sample_rate=0,
                        window_size=0, window_stride=0, window=None,
                        normalize_audio=True, use_filter_pred=True):
        self.data_type = 'audio'

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        if tgt_examples_iter is not None:
            examples_iter = (_join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = _peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (_construct_example_fromlist(ex_values, out_fields)
                        for ex_values in example_values)

        def filter_pred(example):
            if tgt_examples_iter is not None:
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        return out_examples, out_fields, filter_pred
