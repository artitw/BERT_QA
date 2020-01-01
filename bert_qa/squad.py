# Copyright 2019 The BERT-QA Authors. All Rights Reserved.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Copyright 2019 AllenAI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import re
import string
from collections import Counter

from absl import logging
import tensorflow as tf

# pylint: disable=unused-import,g-import-not-at-top,redefined-outer-name,reimported
from . import model_training_utils
from . import bert_modeling as modeling
from . import bert_models
from . import optimization
from . import input_pipeline
from . import model_saving_utils
# word-piece tokenizer based squad_lib
from . import squad_lib as squad_lib_wp
# sentence-piece tokenizer based squad_lib
from . import squad_lib_sp
from . import tokenization
from . import distribution_utils
from . import keras_utils

MODEL_CLASSES = {
    'bert': (modeling.BertConfig, squad_lib_wp, tokenization.FullTokenizer),
    'albert': (modeling.AlbertConfig, squad_lib_sp,
               tokenization.FullSentencePieceTokenizer),
}

# Map string to TensorFlow dtype
DTYPE_MAP = {
    "fp16": tf.float16,
    "bf16": tf.bfloat16,
    "fp32": tf.float32,
}

def get_tf_dtype(dtype="fp32", fp16_implementation="keras"):
  if fp16_implementation == "graph_rewrite":
    # If the graph_rewrite is used, we build the graph with fp32, and let the
    # graph rewrite change ops to fp16.
    return tf.float32
  return DTYPE_MAP[dtype]

def get_loss_scale(loss_scale=None, dtype="fp32", default_for_fp16='dynamic'):
  if loss_scale == "dynamic":
    return loss_scale
  elif loss_scale is not None:
    return float(loss_scale)
  elif dtype == "fp32":
    return 1  # No loss scaling is needed for fp32
  else:
    assert dtype == "fp16"
    return default_for_fp16


def get_num_gpus():
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])

def get_loss_fn(loss_factor=1.0):
  """Gets a loss function for squad task."""

  def _loss_fn(labels, model_outputs):
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs
    return squad_loss_fn(
        start_positions,
        end_positions,
        start_logits,
        end_logits,
        loss_factor=loss_factor)

  return _loss_fn

def squad_loss_fn(start_positions,
                  end_positions,
                  start_logits,
                  end_logits,
                  loss_factor=1.0):
  """Returns sparse categorical crossentropy for start/end logits."""
  start_loss = tf.keras.backend.sparse_categorical_crossentropy(
      start_positions, start_logits, from_logits=True)
  end_loss = tf.keras.backend.sparse_categorical_crossentropy(
      end_positions, end_logits, from_logits=True)

  total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
  total_loss *= loss_factor
  return total_loss


def get_raw_results(predictions, model_type="bert"):
  """Converts multi-replica predictions to RawResult."""
  squad_lib = MODEL_CLASSES[model_type][1]
  for unique_ids, start_logits, end_logits in zip(predictions['unique_ids'],
                                                  predictions['start_logits'],
                                                  predictions['end_logits']):
    for values in zip(unique_ids.numpy(), start_logits.numpy(),
                      end_logits.numpy()):
      yield squad_lib.RawResult(
          unique_id=values[0],
          start_logits=values[1].tolist(),
          end_logits=values[2].tolist())

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class SQuAD(object):

  def __init__(self, 
      model_type="bert",
      do_lower_case=True,
      version_2_with_negative=False,
      distribution_strategy="mirrored",
      hub_module_url=None,
      sp_model_file=None,
      vocab_file="uncased_L-12_H-768_A-12/vocab.txt",
      bert_config_file="uncased_L-12_H-768_A-12/bert_config.json",
      init_checkpoint="uncased_L-12_H-768_A-12/bert_model.ckpt",
      model_dir="model",
      train_batch_size=4,
      learning_rate=8e-5,
      num_train_epochs=2
    ):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.')

    self.strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=distribution_strategy,
      num_gpus=get_num_gpus())

    self.do_lower_case=do_lower_case
    self.version_2_with_negative=version_2_with_negative
    self.model_type=model_type
    self.sp_model_file=sp_model_file
    self.vocab_file=vocab_file
    self.bert_config_file=bert_config_file
    self.init_checkpoint=init_checkpoint
    self.model_dir=model_dir
    self.train_batch_size=train_batch_size
    self.learning_rate=learning_rate
    self.num_train_epochs=num_train_epochs
    self.hub_module_url=hub_module_url

  def get_dataset_fn(self, input_file_pattern, max_seq_length, global_batch_size,
                     is_training):
    """Gets a closure to create a dataset.."""

    def _dataset_fn(ctx=None):
      """Returns tf.data.Dataset for distributed BERT pretraining."""
      batch_size = ctx.get_per_replica_batch_size(
          global_batch_size) if ctx else global_batch_size
      dataset = input_pipeline.create_squad_dataset(
          input_file_pattern,
          max_seq_length,
          batch_size,
          is_training=is_training,
          input_pipeline_context=ctx)
      return dataset

    return _dataset_fn

  def preprocess_training_data(self, 
    squad_data_file="train-v1.1.json",
    max_seq_length=384,
    max_query_length=64,
    doc_stride=128,
    fine_tuning_task_type="squad",
    tokenizer_impl="word_piece",
    input_meta_data_path="squad_v1.1_meta_data",
    train_data_path="squad_v1.1_train.tf_record",
    ):
    """Generates squad training dataset and returns input meta data."""
    self.train_data_path=train_data_path

    if tf.io.gfile.exists(input_meta_data_path):
      with tf.io.gfile.GFile(input_meta_data_path, 'rb') as reader:
        self.input_meta_data = json.loads(reader.read().decode('utf-8'))
        return

    assert squad_data_file
    if tokenizer_impl == "word_piece":
      self.input_meta_data = squad_lib_wp.generate_tf_record_from_json_file(
          squad_data_file, self.vocab_file, self.train_data_path,
          max_seq_length, self.do_lower_case, max_query_length,
          doc_stride, self.version_2_with_negative)
    else:
      assert tokenizer_impl == "sentence_piece"
      self.input_meta_data = squad_lib_sp.generate_tf_record_from_json_file(
          squad_data_file, self.sp_model_file, self.train_data_path, 
          max_seq_length, self.do_lower_case, max_query_length, 
          doc_stride, self.version_2_with_negative)

    with tf.io.gfile.GFile(input_meta_data_path, "w") as writer:
      writer.write(json.dumps(self.input_meta_data, indent=4) + "\n")

  def predict_squad_customized(self, bert_config,
                               predict_tfrecord_path, num_steps):
    """Make predictions using a Bert-based squad model."""
    predict_dataset_fn = self.get_dataset_fn(
        predict_tfrecord_path,
        self.input_meta_data['max_seq_length'],
        self.predict_batch_size,
        is_training=False)
    predict_iterator = iter(
        self.strategy.experimental_distribute_datasets_from_function(
            predict_dataset_fn))

    with self.strategy.scope():
      # Prediction always uses float32, even if training uses mixed precision.
      tf.keras.mixed_precision.experimental.set_policy('float32')
      squad_model, _ = bert_models.squad_model(
          bert_config, input_meta_data['max_seq_length'], float_type=tf.float32)

    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
    logging.info('Restoring checkpoints from %s', checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=squad_model)
    checkpoint.restore(checkpoint_path).expect_partial()

    @tf.function
    def predict_step(iterator):
      """Predicts on distributed devices."""

      def _replicated_step(inputs):
        """Replicated prediction calculation."""
        x, _ = inputs
        unique_ids = x.pop('unique_ids')
        start_logits, end_logits = squad_model(x, training=False)
        return dict(
            unique_ids=unique_ids,
            start_logits=start_logits,
            end_logits=end_logits)

      outputs = self.strategy.experimental_run_v2(
          _replicated_step, args=(next(iterator),))
      return tf.nest.map_structure(self.strategy.experimental_local_results, outputs)

    all_results = []
    for _ in range(num_steps):
      predictions = predict_step(predict_iterator)
      for result in get_raw_results(predictions):
        all_results.append(result)
      if len(all_results) % 100 == 0:
        logging.info('Made predictions for %d records.', len(all_results))
    return all_results


  def fit(self,
          init_checkpoint=None,
          steps_per_loop=200,
          learning_rate=5e-5,
          custom_callbacks=None,
          run_eagerly=False,
          fp16_implementation="keras"):
    """Run bert squad training."""
    if self.strategy:
      logging.info('Training using customized training loop with distribution'
                   ' strategy.')
    # Enables XLA in Session Config. Should not be set for TPU.
    keras_utils.set_config_v2(True)

    use_float16 = get_tf_dtype(fp16_implementation=fp16_implementation) == tf.float16
    if use_float16:
      tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    bert_config = MODEL_CLASSES[self.model_type][0].from_json_file(
        self.bert_config_file)
    epochs = self.num_train_epochs
    num_train_examples = self.input_meta_data['train_data_size']
    max_seq_length = self.input_meta_data['max_seq_length']
    steps_per_epoch = int(num_train_examples / self.train_batch_size)
    warmup_steps = int(epochs * num_train_examples * 0.1 / self.train_batch_size)
    train_input_fn = self.get_dataset_fn(
        self.train_data_path,
        max_seq_length,
        self.train_batch_size,
        is_training=True)

    def _get_squad_model():
      """Get Squad model and optimizer."""
      squad_model, core_model = bert_models.squad_model(
          bert_config,
          max_seq_length,
          float_type=tf.float16 if use_float16 else tf.float32,
          hub_module_url=self.hub_module_url)
      squad_model.optimizer = optimization.create_optimizer(
          learning_rate, steps_per_epoch * epochs, warmup_steps)
      if use_float16:
        # Wraps optimizer with a LossScaleOptimizer. This is done automatically
        # in compile() with the "mixed_float16" policy, but since we do not call
        # compile(), we must wrap the optimizer manually.
        squad_model.optimizer = (
            tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                squad_model.optimizer, loss_scale=get_loss_scale()))
      if fp16_implementation == 'graph_rewrite':
        # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
        # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
        # which will ensure tf.compat.v2.keras.mixed_precision and
        # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
        # up.
        squad_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
            squad_model.optimizer)
      return squad_model, core_model

    # The original BERT model does not scale the loss by
    # 1/num_replicas_in_sync. It could be an accident. So, in order to use
    # the same hyper parameter, we do the same thing here by keeping each
    # replica loss as it is.
    loss_fn = get_loss_fn(loss_factor=1.0)

    model_training_utils.run_customized_training_loop(
        strategy=self.strategy,
        model_fn=_get_squad_model,
        loss_fn=loss_fn,
        model_dir=self.model_dir,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        epochs=epochs,
        train_input_fn=train_input_fn,
        init_checkpoint=init_checkpoint,
        run_eagerly=run_eagerly,
        custom_callbacks=custom_callbacks)


  def predict(self, 
              predict_batch_size=4,
              predict_file="dev-v1.1.json",
              n_best_size=20,
              max_answer_length=30,
              verbose_logging=False
    ):

    self.predict_batch_size=predict_batch_size
    self.predict_file=predict_file

    """Makes predictions for a squad dataset."""
    config_cls, squad_lib, tokenizer_cls = MODEL_CLASSES[self.model_type]
    bert_config = config_cls.from_json_file(self.bert_config_file)
    if tokenizer_cls == tokenization.FullTokenizer:
      tokenizer = tokenizer_cls(
          vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
    else:
      assert tokenizer_cls == tokenization.FullSentencePieceTokenizer
      tokenizer = tokenizer_cls(sp_model_file=self.sp_model_file)
    doc_stride = self.input_meta_data['doc_stride']
    max_query_length = self.input_meta_data['max_query_length']
    # Whether data should be in Ver 2.0 format.
    version_2_with_negative = self.input_meta_data.get('version_2_with_negative',
                                                  False)
    eval_examples = squad_lib.read_squad_examples(
        input_file=self.predict_file,
        is_training=False,
        version_2_with_negative=version_2_with_negative)

    eval_writer = squad_lib.FeatureWriter(
        filename=os.path.join(self.model_dir, 'eval.tf_record'),
        is_training=False)
    eval_features = []

    def _append_feature(feature, is_padding):
      if not is_padding:
        eval_features.append(feature)
      eval_writer.process_feature(feature)

    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
    kwargs = dict(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=self.input_meta_data['max_seq_length'],
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=_append_feature,
        batch_size=self.predict_batch_size)

    # squad_lib_sp requires one more argument 'do_lower_case'.
    if squad_lib == squad_lib_sp:
      kwargs['do_lower_case'] = self.do_lower_case
    dataset_size = squad_lib.convert_examples_to_features(**kwargs)
    eval_writer.close()

    logging.info('***** Running predictions *****')
    logging.info('  Num orig examples = %d', len(eval_examples))
    logging.info('  Num split examples = %d', len(eval_features))
    logging.info('  Batch size = %d', self.predict_batch_size)

    num_steps = int(dataset_size / self.predict_batch_size)
    all_results = self.predict_squad_customized(bert_config,
                                           eval_writer.filename, num_steps)

    output_prediction_file = os.path.join(self.model_dir, 'predictions.json')
    output_nbest_file = os.path.join(self.model_dir, 'nbest_predictions.json')
    output_null_log_odds_file = os.path.join(self.model_dir, 'null_odds.json')

    squad_lib.write_predictions(
        eval_examples,
        eval_features,
        all_results,
        n_best_size,
        max_answer_length,
        self.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose=verbose_logging)


  def evaluate(self, dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

  def export(self, model_export_path):
    """Exports a trained model as a `SavedModel` for inference.

    Args:
      model_export_path: a string specifying the path to the SavedModel directory.
      input_meta_data: dictionary containing meta data about input and model.

    Raises:
      Export path is not specified, got an empty string or None.
    """
    if not model_export_path:
      raise ValueError('Export path is not specified: %s' % model_export_path)
    bert_config = MODEL_CLASSES[self.model_type][0].from_json_file(
        self.bert_config_file)
    squad_model, _ = bert_models.squad_model(
        bert_config, self.input_meta_data['max_seq_length'], float_type=tf.float32)
    model_saving_utils.export_bert_model(
        model_export_path, model=squad_model, checkpoint_dir=self.model_dir)

