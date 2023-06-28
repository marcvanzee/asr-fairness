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


FLAGS = flags.FLAGS

flags.DEFINE_string('db_cache_dir',
                    '/home/marcvanzee/.cache/huggingface/preprocessed_datasets/',
                    'Directory to cache HF datasets after preprocessed')

flags.DEFINE_enum('dataset', 'commonvoice', ['commonvoice', 'voxpopuli'],
                  'Name of dataset to evaluate.')

flags.DEFINE_string('eval_results_dir',
                    './eval_results',
                    'Directory to store eval results tsv files')

flags.DEFINE_list('model_sizes',
                  ['tiny', 'base', 'small', 'medium', 'large', 'large-v2'],
                  'Whisper model sizes to evaluate.')


# TODO: parallel dataset mapping sometimes gets stuck, investigate why.
_NUM_CORES = 1  # os.cpu_count

# Always evaluate on 'test'.
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


def get_sentence_key():
  return {
    'commonvoice': 'sentence',
    'voxpopuli': 'normalized_text'
  }[FLAGS.dataset]


def get_dataset(dataset_name, split, lang, batch_size=8):
  # Note: we can use larger batch sizes for smaller models, but batching
  # the dataset takes a lot of time so we just use the same batch size
  # everywhere and cache the preprocessed dataset.

  def preprocess_audio(e):
    audio = whisper.pad_or_trim(e['audio']['array'].flatten())
    mels = whisper.log_mel_spectrogram(audio)
    key = 'sentence' if dataset_name == 'commonvoice' else 'normalized_text'
    return { 'mels': mels, 'text':  e[key] }

  def stack_ds(examples):
    return {
      **{'mels': torch.stack([examples['mels']])},
      **{k: [v] for k, v in examples.items()}
    }
  
  # Legacy, remove.
  dataset_cache_name = dataset_name
  if dataset_name == 'commonvoice':
    dataset_cache_name = 'common_voice_9_0'

  db_cache_path = os.path.join(FLAGS.db_cache_dir,
                                 f'{dataset_cache_name}_{lang}_{split}')

  if os.path.exists(db_cache_path):
    print(f'Loading db cache from: {db_cache_path}')
    ds = load_from_disk(db_cache_path)
  else:
    if os.path.exists(db_cache_path + '_nobatch'):  # Legacy: remove
      print('Loading legacy preprocessed dataset')
      ds = load_from_disk(db_cache_path + '_nobatch')
    else:
      print('Downloading dataset')
      dataset = {
        'commonvoice': 'mozilla-foundation/common_voice_9_0',
        'voxpopuli': 'facebook/voxpopuli'
      }[dataset_name]
      ds = load_dataset(dataset, lang, split=split, use_auth_token=True,
                        num_proc=_NUM_CORES)
      ds = ds.cast_column("audio", Audio(sampling_rate=16000)).with_format("torch")
      ds = ds.map(preprocess_audio, num_proc=_NUM_CORES)
    ds = ds.map(stack_ds, batched=True, batch_size=batch_size)
    ds.save_to_disk(db_cache_path)
  
  return ds


def eval(model, ds, lang):
  options = whisper.DecodingOptions(language=get_lang(lang), without_timestamps=True)
  results = collections.defaultdict(list)

  keys_to_map = {
    'commonvoice': ['age', 'gender', 'accent', 'locale'],
    'voxpopuli': ['gender', 'speaker_id', 'accent'],
  }[FLAGS.dataset]

  for batch in tqdm(ds):
    outputs = model.decode(batch['mels'].to('cuda'), options)
    results['reference_original'].extend(batch[get_sentence_key()])
    results['inferred_original'].extend([x.text for x in outputs])
    for key in keys_to_map:
      results[key].extend(batch[key])
  
  normalizer = EnglishTextNormalizer() if lang == 'en' else BasicTextNormalizer()
  results['reference'] = [normalizer(x) for x in results['reference_original']]
  results['inferred'] = [normalizer(x) for x in results['inferred_original']]

  return results


def save_results(result_path, results):
  keys = results.keys()
  print(f'Saving {len(results.keys()[0])} results to {result_path}.')

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


def get_model_type(model_size, lang):
  if lang == 'en' and 'large' not in model_size:
    return model_size + '.en'
  else:
    return model_size


def main(_):
  langs_to_eval = get_langs_to_eval(FLAGS.dataset)
  print('Evaluating dataset', FLAGS.dataset)
  print(f'Evaluating {len(langs_to_eval)} languages: {langs_to_eval}')
  print(f'Evaluating Whisper models sizes: {FLAGS.model_sizes}')

  # First iterate over model size so we can keep the same model in memory for
  # all languages -- dataset loading is quite fast once it is cached.
  prev_model_type = None
  model = None
  for model_size in FLAGS.model_sizes:
    print('=== Evaluating model size', model_size)
    for lang in langs_to_eval:
      print('=== Evaluating language', lang)
      result_path = f'./eval_results/whisper_{model_size}_{FLAGS.dataset}_{lang}_{_SPLIT}.tsv'
      if os.path.exists(result_path):
        print(f'Results exist already -- skipping:', result_path)
        continue

      if (model_type := get_model_type(model_size, lang)) != prev_model_type:
        # Switch the model if it is different from the previous iteration.
        print('=== Loading new model', model_type)
        model = whisper.load_model(model_type)
        prev_model_type = model_type

      ds = get_dataset(FLAGS.dataset, _SPLIT, lang)
      results = eval(model, ds, lang)
      save_results(result_path, results)


if __name__ == '__main__':
  app.run(main)
