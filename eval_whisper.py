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
    'ab', 'ar', 'as', 'az', 'ba', 'bas', 'be', 'bg', 'bn', 'br', 'ca', 'ckb',
    'cnh', 'cs', 'cv', 'cy', 'da', 'de', 'dv', 'el', 'en', 'eo', 'es', 'et',
    'eu', 'fa', 'fi', 'fr', 'fy-NL', 'ga-IE', 'gl', 'gn', 'ha', 'hi', 'hsb',
    'hu', 'hy-AM', 'ia', 'id', 'ig', 'it', 'ja', 'ka', 'kab', 'kk', 'kmr', 'ky',
    'lg', 'lt', 'lv', 'mdf', 'mhr', 'mk', 'ml', 'mn', 'mr', 'mt', 'myv',
    'nan-tw', 'nl', 'nn-NO', 'or', 'pa-IN', 'pl', 'pt', 'rm-sursilv',
    'rm-vallader', 'ro', 'ru', 'rw', 'sah', 'sat', 'sk', 'sl', 'sr', 'sv-SE',
    'sw', 'ta', 'th', 'tig', 'tok', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi',
    'vot', 'yue', 'zh-CN', 'zh-HK', 'zh-TW'
]

_VOXPOPULI_LANGS = [
    'lt', 'en', 'de', 'fr', 'es', 'pl', 'it', 'ro', 'hu', 'cs', 'nl', 'fi', 'hr',
    'sk', 'sl', 'et',
]

 
def get_batch_size(model_size):
  return {
      'tiny': 64,
      'base': 64,
      'small': 32,
      'medium': 16,
      'large': 16,
      'large-v2': 8,
  }[model_size]


def get_model(model_size, lang):
  model_type = model_size
  if 'large' not in model_size and get_lang(lang) == 'en':
    model_type += ".en"
  return whisper.load_model(model_type)


def get_sentence_key():
  return {
    'commonvoice': 'sentence',
    'voxpopuli': 'normalized_text'
  }[FLAGS.dataset]


def get_dataset(dataset_name, split, lang, batch_size):
  def preprocess_audio(e):
    audio = whisper.pad_or_trim(e['audio']['array'].flatten())
    mels = whisper.log_mel_spectrogram(audio)
    return { 'mels': mels, 'text':  e[get_sentence_key()] }

  def stack_ds(examples):
    return {
      **{'mels': torch.stack([examples['mels']])},
      **{k: [v] for k, v in examples.items()}
    }

  # Try retrieving two types of preprocessed datasets: with and without batching.
  dataset_name.replace('/', '___')
  db_cache_prefix = os.path.join(FLAGS.db_cache_dir,
                                 f'{dataset_name}_{lang}_{split}')
  db_unbatched_cache_path = db_cache_prefix + '_nobatch'
  db_batched_cache_path = db_cache_prefix + f'_bs_{batch_size}'

  do_batch = True

  if os.path.exists(db_batched_cache_path):
    print(f'Loading batched db cache from: {db_batched_cache_path}')
    ds = load_from_disk(db_batched_cache_path)
    do_batch = False
  elif os.path.exists(db_unbatched_cache_path):
    print(f'Loading unbatched db cache from {db_unbatched_cache_path}')
    ds = load_from_disk(db_unbatched_cache_path)
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
    ds.save_to_disk(db_unbatched_cache_path)
    
  if do_batch:
    print(f'Batching dataset (batch size = {batch_size})')
    ds = ds.map(stack_ds, batched=True, batch_size=batch_size)
    ds.save_to_disk(db_batched_cache_path)
  
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


def main(_):
  langs_to_eval = get_langs_to_eval(FLAGS.dataset)
  print('Evaluating dataset', FLAGS.dataset)
  print(f'Evaluating {len(langs_to_eval)} languages: {langs_to_eval}')
  print(f'Evaluating Whisper models sizes: {FLAGS.model_sizes}')
  
  for lang in langs_to_eval:
    print('=== Language', lang)
    for model_size in FLAGS.model_sizes:
      result_path = f'./eval_results/whisper_{model_size}_{FLAGS.dataset}_{lang}_{_SPLIT}.tsv'
      if os.path.exists(result_path):
        print(f'Results exist already -- skipping:', result_path)
        continue
      model = get_model(model_size, lang)
      batch_size = get_batch_size(model_size)
      print(f'=== Whisper model {model_size}, batch size = {batch_size}')

      ds = get_dataset(FLAGS.dataset, _SPLIT, lang, batch_size)
      results = eval(model, ds, lang)
      save_results(result_path, results)


if __name__ == '__main__':
  app.run(main)
