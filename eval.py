import torch
from datasets import load_dataset, Audio, load_from_disk
import whisper
import os
import jiwer
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
import pandas as pd
from tqdm import tqdm
import collections
import csv
import re
from absl import app
from absl import flags
from transformers import Wav2Vec2ForCTC, AutoProcessor
import dataclasses


FLAGS = flags.FLAGS

flags.DEFINE_string('db_cache_dir',
                    '/home/marcvanzee/.cache/huggingface/preprocessed_datasets/',
                    'Directory to cache HF datasets after preprocessed')

flags.DEFINE_enum('dataset', 'commonvoice', ['commonvoice', 'voxpopuli'],
                  'Name of dataset to evaluate.')

flags.DEFINE_string('eval_results_dir',
                    './eval_results',
                    'Directory to store eval results tsv files')

flags.DEFINE_enum('model', 'whisper', ['whisper', 'mms'],
                  'Model to evaluate.')

flags.DEFINE_list('whisper_model_sizes',
                  ['tiny', 'base', 'small', 'medium', 'large', 'large-v2'],
                  'Whisper model sizes to evaluate.')

flags.DEFINE_list('mms_model_sizes',
                  ['mms-1b-fl102', 'mms-1b-l1107', 'mmb-1b-all'],
                  'MMS model sizes to evaluate.')


# TODO: parallel dataset mapping sometimes gets stuck, investigate why.
_NUM_CORES = 1  # os.cpu_count

# We always evaluate on the test split.
_SPLIT = 'test'


_WHISPER_LANGS_SUPPORTED = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs',
    'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi',
    'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'hi', 'hr', 'ht', 'hu', 'hy', 'id',
    'is', 'it', 'iw', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb',
    'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
    'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
    'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw',
    'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi',
    'yi', 'yo', 'zh']

_MMS_LANGS_SUPPORTED = [
  # TODO: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
  # MMS uses ISO-639-3 while the datasets use 639-1: convert.
]

_COMMONVOICE9_LANGS = [
    'ab', 'ar', 'as', 'az', 'ba', 'bas', 'be', 'bg', 'bn', 'br', 'ca', 
    'cnh', 'cs', 'cv', 'cy', 'da', 'de', 'dv', 'el', 'eo', 'es', 'et', 'ckb',
    'eu', 'fa', 'fi', 'fr', 'fy-NL', 'ga-IE', 'gl', 'gn', 'ha', 'hi', 'hsb',
    'hu', 'hy-AM', 'ia', 'id', 'ig', 'it', 'ja', 'ka', 'kab', 'kk', 'kmr', 'ky',
    'lg', 'lt', 'lv', 'mdf', 'mhr', 'mk', 'ml', 'mn', 'mr', 'mt', 'myv',
    'nan-tw', 'nl', 'nn-NO', 'or', 'pa-IN', 'pl', 'pt', 'rm-sursilv',
    'rm-vallader', 'ro', 'ru', 'rw', 'sah', 'sat', 'sk', 'sl', 'sr', 'sv-SE',
    'sw', 'ta', 'th', 'tig', 'tok', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi',
    'vot', 'yue', 'zh-CN', 'zh-HK', 'zh-TW', 'en'
]

_VOXPOPULI_LANGS = [
    'lt', 'de', 'fr', 'es', 'pl', 'it', 'ro', 'hu', 'cs', 'nl', 'fi', 'hr',
    'sk', 'sl', 'et', 'en'
]


@dataclasses.dataclass
class WhisperModel:
  model: whisper.model.Whisper
  options: whisper.decoding.DecodingOptions
  name: str = 'whisper'

  @staticmethod
  def preprocess_audio(x):
    # Note: this is a staticmethod since Arrow datasets cannot serialize the
    # Whisper object, os we cannot pass an instance of this class since the
    # cached dataset will not be properly hased.
    return whisper.log_mel_spectrogram(whisper.pad_or_trim(x))
  
  def decode(self, batch):
    return self.model.decode(batch['inputs'].to('cuda'), self.options)


@dataclasses.dataclass
class MMSModel:
  model: Wav2Vec2ForCTC
  name: str = 'mms'

  @staticmethod
  def preprocess_audio(processor, x):
    return processor(x, sampling_rate=16_000, return_tensors='pt')
  
  def decode(self, batch):
    assert len(batch) == 1, 'MMS requires a batch size of 1'
    inputs = {k: v.to('cuda') for k, v in batch['inputs'][0].items()}
    with torch.no_grad():
      outputs = self.model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    return self.processor.decode(ids)


def get_protected_attrs(dataset):
  return {
    'commonvoice': ['age', 'gender', 'accent', 'locale'],
    'voxpopuli': ['gender', 'accent'],
  }[dataset]


def get_batch_size(model):
  return {
    # Note: we can use larger batch sizes for smaller models, but batching
    # the dataset takes a lot of time so we just use the same batch size
    # everywhere and cache the preprocessed dataset.
    'whisper': 8,
    # The Huggingface interface to MMS only seems to work for single examples.
    'mms': 1,
  }[model.name]


def get_batches(dataset, lang, audio_preprocess_fn, batch_size):
  def preprocess(hf_name, col_reference, cols_to_keep, cols_to_remove):
    def preprocess_fn(examples):
      inputs = [audio_preprocess_fn(x['array'].flatten()) for x in examples['audio']]
      return { 
          **{'inputs': [inputs],
             'reference_original': [examples[col_reference]]},
          **{k: [examples[k]] for k in cols_to_keep }
      }
    print('Downloading dataset', hf_name)
    ds = load_dataset(hf_name, lang, split=_SPLIT, use_auth_token=True,
                      num_proc=_NUM_CORES)
    ds = ds.cast_column('audio', Audio(sampling_rate=16000)).with_format('torch')
    ds = ds.map(preprocess_fn, num_proc=_NUM_CORES,
            remove_columns=cols_to_remove, batched=True,
            batch_size=batch_size)
    return ds
    
  dataset_cache_name = dataset
  if dataset == 'commonvoice':  # Legacy, remove.
    dataset_cache_name = 'common_voice_9_0'
  db_cache_path = os.path.join(FLAGS.db_cache_dir,
                                 f'{dataset_cache_name}_{lang}_{_SPLIT}')

  if os.path.exists(db_cache_path):
    print(f'Loading db cache from: {db_cache_path}')
    ds = load_from_disk(db_cache_path)
  else:
    if dataset == 'commonvoice':
      ds = preprocess(
        hf_name='mozilla-foundation/common_voice_9_0',
        col_reference='sentence',
        cols_to_keep=get_protected_attrs(dataset),
        cols_to_remove=['client_id', 'path', 'audio', 'up_votes', 'down_votes',
                        'segment', 'sentence'])
    else:
      ds = preprocess(
        hf_name='facebook/voxpopuli',
        col_reference='normalized_text',
        cols_to_keep=get_protected_attrs(dataset),
        cols_to_remove=['audio_id', 'audio', 'raw_text', 'language', 'speaker_id', 
                      'is_gold_transcript', 'normalized_text'])
    
    ds.save_to_disk(db_cache_path)
  
  return ds


def get_text_normalizer(model, lang):
  normalize_fn = {
    'whisper': EnglishTextNormalizer() if lang == 'en' else BasicTextNormalizer(),
    'mms': lambda x: x
  }[model.name]
  return lambda xs: list(map(normalize_fn, xs))


def save_results(result_path, results):
  keys = list(results.keys())
  print(f'Saving {len(keys[0])} results to {result_path}.')

  with open(result_path, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(keys)
    writer.writerows(zip(*[results[key] for key in keys]))


def get_lang(code):
  return re.split('_|-', code)[0] if code else None


def get_langs_to_eval(dataset_name):
  dataset_langs = {
    'commonvoice': _COMMONVOICE9_LANGS,
    'voxpopuli': _VOXPOPULI_LANGS
  }[dataset_name]
  langs = [x for x in dataset_langs if get_lang(x) in _WHISPER_LANGS_SUPPORTED]
  return langs


def get_model_id(model_name, model_size, lang):
  use_en_model = lang == 'en' and 'large' not in model_size
  return {
    'mms': 'facebook/' + model_size,
    'whisper': model_size + '.en' if use_en_model else model_size
  }[model_name]
  

def get_model_sizes(model_name):
  return {
    'whisper': FLAGS.whisper_model_sizes,
    'mms': FLAGS.mms_model_sizes
  }[model_name]


def load_model_and_preprocessor(model_name, model_id, lang):
  print('=== Loading new model', model_name, model_id)
  if model_name == 'whisper':
    model = WhisperModel(
      model=whisper.load_model(model_id),
      options=whisper.DecodingOptions(language=get_lang(lang),
                                      without_timestamps=True))
    preprocessor = WhisperModel.preprocess_audio
  elif model_name == 'mms':
    model = MMSModel(
      model=Wav2Vec2ForCTC.from_pretrained(model_id).to('cuda'),
      processor=AutoProcessor.from_pretrained(model_id)
    )
    preprocessor = MMSModel.preprocess_audio

  return model, preprocessor
  

def main(_):
  model_name = FLAGS.model
  model_sizes = get_model_sizes(model_name)
  dataset = FLAGS.dataset
  langs_to_eval = get_langs_to_eval(dataset)

  print( ' -------------- Evaluation settings --------------------------------')
  print( '| Model', FLAGS.model)
  print(f'| Models sizes: {model_sizes}')
  print( '| Dataset', dataset)
  print(f'| {len(langs_to_eval)} languages: {langs_to_eval}')
  print( '|-------------------------------------------------------------------')

  model = None
  # Keep track of the model id so we can reuse the model it if possible.
  model_id = None
  prev_model_id = None

  # First iterate over model size so we can keep the same model in memory for
  # all languages -- dataset loading is quite fast once it is cached.
  for model_size in model_sizes:
    print('=== Evaluating model size', model_size)
    for lang in langs_to_eval:
      print('=== Evaluating language', lang)
      result_path = os.path.join(
        FLAGS.eval_results_dir,
        f'{model_name}_{model_size}_{FLAGS.dataset}_{lang}_{_SPLIT}.tsv')
      print(result_path)
      if os.path.exists(result_path):
        print(f'Results exist already -- skipping:', result_path)
        continue

      model_id = get_model_id(model_name, model_size, lang)
      if model_id != prev_model_id:
        # Switch the model if it is different from the previous iteration.
        model, preprocess_fn = load_model_and_preprocessor(FLAGS.model, model_id, lang)
        prev_model_id = model_id

      results = collections.defaultdict(list)
      normalize = get_text_normalizer(model, lang)

      batch_size = get_batch_size(model)
      for batch in tqdm(get_batches(dataset, lang, preprocess_fn, batch_size)):
        outputs = model.decode(batch)
        results['reference_original'].extend(batch['reference_original'])
        results['inferred_original'].extend([x.text for x in outputs])
        results['reference'].extend(normalize(results['reference_original']))
        results['inferred'].extend(normalize(results['inferred_original']))
        for key in get_protected_attrs(dataset):
          results[key].extend(batch[key])

      save_results(result_path, results)


if __name__ == '__main__':
  app.run(main)
