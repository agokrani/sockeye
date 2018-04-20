"""
Scoring CLI
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from typing import Any, cast, Optional, Dict, List, Tuple
import decimal

import mxnet as mx
import numpy as np

from . import arguments
from . import data_io
from . import training
from . import vocab
from . import constants as C
from . import model
from . import utils
from . import loss
from .log import setup_main_logger
from .utils import acquire_gpus, get_num_gpus, log_basic_info, grouper

import pdb

logger = setup_main_logger(__name__, file_logging=False, console=True)

def main():
    params = argparse.ArgumentParser(description='Scoring CLI')
    arguments.add_scoring_cli_args(params)
    args = params.parse_args()

    log_basic_info(args)

    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)

    source_vocab, target_vocab = load_vocabs(args.model)
    logger.info("Successfully loaded source and target vocabularies.")
    """
    source_vocabs, target_vocab = vocab.load_or_create_vocabs(
    source_paths=["/u/gokrani/Desktop/data_sockeye/trainingset_sockeye/Trainingset.de.en.de_200.gz"],
    target_path="/u/gokrani/Desktop/data_sockeye/trainingset_sockeye/Trainingset.de.en.en_200.gz",
    source_vocab_paths=["/work/smt2/gokrani/experiments/sockeye/code_change/vocab.src.0.json"],
    target_vocab_path="/work/smt2/gokrani/experiments/sockeye/code_change/vocab.trg.json",
    shared_vocab=False,
    num_words_source=50000,
    num_words_target=50000,
    word_min_count_source=1,
    word_min_count_target=1)"""
    """models, source_vocabs, target_vocab = scoring.load_models(context=mx.cpu(),
                                                               max_input_len=60,
                                                               model_folders=["/work/smt2/gokrani/experiments/sockeye/code_change"])"""

    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    max_seq_len_source, max_seq_len_target = args.max_seq_len
    utils.check_condition(args.fill_up is None, "Random sampling can cause scoring of single sentence repeatedly. Please turn off fill up.")

    #shared_vocab = False by default
    #Batch by words = False by default

    score_iter, config_data, data_info = data_io.get_scoring_data_iters(
        sources=[args.source],
        target=os.path.abspath(args.target),
        source_vocabs=[source_vocab],
        target_vocab=target_vocab,
        source_vocab_paths=os.path.join(args.model[0], C.VOCAB_SRC_NAME % 0),
        target_vocab_path=os.path.join(args.model[0], C.VOCAB_TRG_NAME),
        shared_vocab=False,
        batch_size = 64,
        batch_by_words=False,
        batch_num_devices=batch_num_devices,
        fill_up=None,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target,
        bucketing=not args.no_bucketing,
        bucket_width=10)

    scorer = load_models(context=context,
                max_input_len=max_seq_len_source,
                model_folder=args.model[0],
                score_iter=score_iter)


    scorer = LogProbabilityScorer(model=scorer, source_vocabs=[source_vocab], target_vocab=target_vocab)
    scorer.get_scored_dataset(train_iter=score_iter)


class ScoringModel(model.SockeyeModel):
    """
            ScoringModel is a SockeyeModel that computes the forward pass over all the training sequences.
            :param config: Configuration object holding details about the model.
            :param params_fname: File with model parameters.
            :param context: MXNet context to bind modules to.
            :param score_iter: data iterator for data set to be scored.

    """

    def __init__(self,
                 config: model.ModelConfig,
                 params_fname: str,
                 context: mx.context.Context,
                 model_dir: str,
                 score_iter: data_io.BaseParallelSampleIter,
                 bucketing: bool):
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.model_dir = model_dir
        self.bucketing = bucketing
        self.module = self._initialize(score_iter=score_iter)

    def _initialize(self, score_iter: data_io.BaseParallelSampleIter):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        #utils.check_condition(score_iter.pad_id == C.PAD_ID == 0, "pad id should be 0")
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]

        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        model_loss = loss.get_loss(self.config.config_loss)

        data_names = [x[0] for x in score_iter.provide_data]
        label_names = [x[0] for x in score_iter.provide_label]




        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (source_encoded_length, batch_size, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # target_decoded: (batch_size * target_seq_len, rnn_num_hidden)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(target_decoded)

            probs = model_loss.get_loss(logits, labels)

            return mx.sym.Group(probs), data_names, label_names



        #symbol = mx.sym.load(os.path.join(modeldir, 'symbol.json'))

        if self.bucketing:
            default_bucket_key = (self.config.config_data.max_seq_len_source, self.config.config_data.max_seq_len_source)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                                 logger=logger,
                                                 default_bucket_key=default_bucket_key,
                                                 context=self.context)

        symbol, _, __ = sym_gen(score_iter.buckets[-1])

        logger.info("No bucketing. Unrolled to (%d,%d)",
                    self.config.config_data.max_seq_len_source, self.config.config_data.max_seq_len_target)

        return mx.mod.Module(symbol=symbol,
                             data_names=data_names,
                             label_names=label_names,
                             logger=logger,
                             context=self.context)


def _setup_context(args, exit_stack):
    if args.use_cpu:
        context = mx.cpu()
    else:
        num_gpus = get_num_gpus()
        check_condition(num_gpus >= 1,
                        "No GPUs found, consider running on the CPU with --use-cpu "
                        "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi "
                        "binary isn't on the path).")
        check_condition(len(args.device_ids) == 1, "cannot run on multiple devices for now")
        gpu_id = args.device_ids[0]
        if args.disable_device_locking:
            if gpu_id < 0:
                # without locking and a negative device id we just take the first device
                gpu_id = 0
        else:
            gpu_ids = exit_stack.enter_context(acquire_gpus([gpu_id], lock_dir=args.lock_dir))
            gpu_id = gpu_ids[0]

        context = mx.gpu(gpu_id)
    return context

def load_vocabs(model_folder: str):
    logger.info("Loading source and target vocabularies.")
    vocab_source = vocab.vocab_from_json(os.path.join(model_folder[0], C.VOCAB_SRC_NAME % 0))
    vocab_target = vocab.vocab_from_json(os.path.join(model_folder[0], C.VOCAB_TRG_NAME))
    return vocab_source, vocab_target


def load_models(context: mx.context.Context,
                max_input_len: Optional[int],
                model_folder: str,
                score_iter: data_io.BaseParallelSampleIter,
                checkpoint: Optional[int] = None,
                bucketing: bool = True) -> ScoringModel:

    model_config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))

    params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint if checkpoint else C.PARAMS_BEST_NAME)
    utils.check_condition(os.path.exists(params_fname), "No model parameter file found under %s. " % params_fname)


    if max_input_len is not None:
        utils.check_condition(max_input_len <= model_config.config_data.max_seq_len_source,
                              "Model only supports a maximum length of %d" % model_config.config_data.max_seq_len_source)



    scorer = ScoringModel(config=model_config,
                           params_fname=params_fname,
                           context=context,
                           model_dir=model_folder,
                           score_iter=score_iter,
                           bucketing=bucketing)

    scorer.load_params_from_file(params_fname)  # sets scoring.params


    #scoring.module.bind(data_shapes=score_iter.provide_data, label_shapes=score_iter.provide_label)
    scorer.module.bind(data_shapes=score_iter.provide_data,
                        label_shapes=score_iter.provide_label)
    scorer.module.set_params(arg_params=scorer.params, aux_params=scorer.aux_params)
    logger.info('Loaded params from "%s"', params_fname)

    return scorer


# class to store and write sentences with their source and target side along with the log probability score
# computed by LogProbabilityScorer
class ScoredSentence:
    """
            Scored sentence stores the source sentence and target sentence along with their computed log probability scores.

            :param source: soruce sentence.
            :param target: target sentence.
            :param score: computed log probability score for the sentence."""
    def __init__(self,
                 source: str,
                 target: str,
                    score: float):
            self.source = source
            self.target = target
            self.score = score

    def save_sentence(self, fp):
        """
            writes a single sentence along with their score to the file.

            :param fp: file pointer to the output file."""
        fp.write(self.source + "\n")
        fp.write(self.target + "\n")
        fp.write(str(decimal.Decimal(np.float(self.score))) + "\n")

# Class that performs forward pass over whole training data and computes the score using the
# log probabilities computed in the forward pass


class LogProbabilityScorer:
    """
        LogProbabilityScorer computes the score of forward pass from the already trained model for certain epochs
        and returns the scored dataset to a file.

        :param model: trained model to be used for forward pass computation.
        :param source_vocab: Source vocabulary for already trained model. Should be same as the one used at training time.
        :param target_vocab: Target vocabulary for already trained model.Should be same as the one used at training time."""

    def __init__(self,
                 model: ScoringModel,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab)-> None:
        self.model = model
        self.source_vocabs = source_vocabs
        self.source_vocabs_inv = [vocab.reverse_vocab(source_vocab) for source_vocab in self.source_vocabs]
        self.vocab_target = target_vocab
        self.vocab_target_inv = vocab.reverse_vocab(self.vocab_target)
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}
        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        self.scored_dataset_dir_name = os.path.join(self.model.model_dir, C.SCORED_DATASET_DIR_NAME)
        if not os.path.exists(self.scored_dataset_dir_name):
            os.mkdir(self.scored_dataset_dir_name)
        self.scored_dataset_file_pointer = utils.smart_open(os.path.join(self.scored_dataset_dir_name, C.SCORED_DATASET_FILE_NAME), "wt")

    def get_scored_dataset(self,
                      train_iter: data_io.BaseParallelSampleIter) -> None:
        """
           computes the forward pass, log probability scores and writes the scored dataset to the file.

           :param train_iter: iterator over training data.
           """


        logger.info("Starting score computation for the dataset")
        batch_count = 0;
        while train_iter.iter_next():
            logger.info("Computing scores for batch[%d]" % batch_count)
            batch = train_iter.next()

            forward_pass_outputs = self._run_forward_pass(batch)
            scores = self._compute_scores(forward_pass_outputs, batch.label[0])

            logger.info("Converting source and target from ids to tokens and aligning it with scores for batch[%d]" % batch_count)
            scored_sentences = self._ids2tokens_with_score_alignment(batch, scores)

            logger.info(
                "writing the scored batch[%d] to the file" % batch_count)
            self._save_scored_sentences(scored_sentences)

            batch_count += 1
        self.scored_dataset_file_pointer.close()
        logger.info("score computation completed. Scored dataset has been written to the file.")




    def _run_forward_pass(self, batch: mx.io.DataBatch) -> mx.ndarray:
        """
            computes the forward pass for each batch and return the results.

            :param batch: data batch for the computation of forward pass.
            :return: The results for the computation of forward pass.
            """

        #states = self.model.run_encoder(batch, batch.data[0].shape[1])

        self.model.module.forward(batch)

        return self.model.module.get_outputs()[0].reshape((batch.label[0].shape[0], batch.label[0].shape[1],
                                                                       self.model.config.vocab_target_size))

    def _ids2tokens_with_score_alignment(self, batch: mx.io.DataBatch, scores: mx.ndarray):
        """
            convert ids in the iterator to the corresponding tokens of the source and target language in the source and target vocabularies.

            :param batch: data batch
            :param scores: log probability scores.
            :return: The object of ScoredSentence which contains sentence in source language, target language and the computed log probability score.
            """
        scored_sentences = []
        for i in range(batch.data[0].shape[0]):
            source_token_ids = batch.data[0][i]
            source_tokens = [self.source_vocabs_inv[0][source_token_id.asscalar()] for source_token_id in source_token_ids]
            source_sentence = C.TOKEN_SEPARATOR.join(source_token for source_id, source_token in zip(source_token_ids, source_tokens) if source_id.asscalar() not in self.stop_ids)

            target_token_ids = batch.label[0][i]
            target_tokens = [self.vocab_target_inv[target_token_id.asscalar()] for target_token_id in target_token_ids]
            target_sentence = C.TOKEN_SEPARATOR.join(target_token for target_id, target_token in zip(target_token_ids, target_tokens) if target_id.asscalar() not in self.stop_ids)
            if len(source_sentence) > 0:
                scored_sentences.append(ScoredSentence(source_sentence, target_sentence, score=scores[i].asscalar()))


        return scored_sentences


    def _compute_scores(self, forward_pass_outputs: mx.ndarray, labels: mx.ndarray):
        """
            for each word in the sentence we have a probability distribution over total number of dictionary words. From
            that probability distributions for each word, we select the probability for the word which is correct target word
            and we sum the probabilities for each word in the sentence to get one score for each sentence. All the
            probabilities are log probabilities.

            :param forward_pass_outputs: forward pass outputs for each data batch.
            :param labels: correct target language words.
            :return: list of log probability scores for each sentence in the data batch.
            """

        scores = np.empty(0)
        for i in range(forward_pass_outputs.shape[0]):
            sum = 0
            for j in range(forward_pass_outputs.shape[1]):
                sum += forward_pass_outputs[i][j][labels[i][j]]
            scores = np.append(scores, sum)
        return scores

    def _save_scored_sentences(self, scored_sentences: List[ScoredSentence]):
        """
            writes each sentence in  the batch along with their score to the file.

            :param scored_sentences: list of ScoredSentences in a data batch. """
        for i in range(len(scored_sentences)):
            scored_sentences[i].save_sentence(self.scored_dataset_file_pointer)



if __name__ == '__main__':
    main()