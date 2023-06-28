import torch
from datasets import load_dataset, Audio, load_from_disk
import whisper
import numpy as np
import os
import jiwer
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
import pandas as pd
from tqdm import tqdm
import collections
import math
import csv
import re

_db_cache_dir = '/home/marcvanzee/.cache/huggingface/preprocessed_datasets/'
_num_cores = 1  # os.cpu_count()ds

print('CPU cores', _num_cores)


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
  model_type = model_size + ".en" if get_lang(lang) == "en" else model_size
  model = whisper.load_model(model_type)
  return model

def get_dataset(dataset_name, split, lang, batch_size):
  def preprocess_audio(e):
    audio = whisper.pad_or_trim(e['audio']['array'].flatten())
    mels = whisper.log_mel_spectrogram(audio)
    return { 'mels': mels, 'text':  e['sentence'] }

  def stack_ds(examples):
    return {
      **{'mels': torch.stack([examples['mels']])},
      **{k: [v] for k, v in examples.items()}
    }

  # Try retrieving two types of preprocessed datasets: with and without batching.
  dataset_name.replace('/', '___')
  db_cache_prefix = os.path.join(_db_cache_dir, f'{dataset_name}_{lang}_{split}')
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
    ds = load_dataset(dataset_name, lang, split=split, use_auth_token=True, num_proc=_num_cores)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)).with_format("torch")
    ds = ds.map(preprocess_audio, num_proc=_num_cores)
    ds.save_to_disk(db_unbatched_cache_path)
    
  if do_batch:
    print(f'Batching dataset (batch size = {batch_size})')
    ds = ds.map(stack_ds, batched=True, batch_size=batch_size)
    ds.save_to_disk(db_batched_cache_path)
  
  return ds

def add_results(results, batch, outputs):
  for key in ['age', 'gender', 'accent', 'locale']:
    results[key].extend(batch[key])

  results['reference_original'].extend(batch['sentence'])
  results['inferred_original'].extend([x.text for x in outputs])
  return results
  
def eval(model, ds, lang):
  options = whisper.DecodingOptions(language=get_lang(lang), without_timestamps=True)
  results = collections.defaultdict(list)

  for batch in tqdm(ds):
    outputs = model.decode(batch['mels'].to("cuda"), options)
    results = add_results(results, batch, outputs)
  
  normalizer = EnglishTextNormalizer() if lang == 'en' else BasicTextNormalizer()

  results['reference'] = [normalizer(x) for x in results['reference_original']]
  results['inferred'] = [normalizer(x) for x in results['inferred_original']]

  wer = lambda ref, hyp: -math.inf if not ref.strip() else jiwer.wer(ref, hyp)

  results['wer'] = [
    wer(*xs) for xs in zip(results['reference'],results['inferred'])
  ]

  wers = list(filter(lambda x: x != -math.inf, results['wer']))
  avg_wer = sum(wers) / len(wers)
  print(f"WER: {avg_wer * 100:.1f}% (skipped {len(wers)-len(results['wer'])})")

  return results

def save_results(result_path, results):
  print(f'Saving {len(results)} results to {result_path}.')

  with open(result_path, 'w') as f:
    keys = results.keys()
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(keys)
    writer.writerows(zip(*[results[key] for key in keys]))

def get_lang(code):
  return re.split('_|-', code)[0] if code else None

_whisper_langs_supported = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs',
    'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi',
    'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'hi', 'hr', 'ht', 'hu', 'hy', 'id',
    'is', 'it', 'iw', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb',
    'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
    'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
    'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw',
    'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi',
    'yi', 'yo', 'zh']

_cv_langs = [
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

langs_to_eval = [x for x in _cv_langs if get_lang(x) in _whisper_langs_supported]

print(f'Evaluating {len(langs_to_eval)} languages: {langs_to_eval}')


dataset_name = 'mozilla-foundation/common_voice_9_0'
eval_dir = './eval_results'
model_sizes = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2']
split = 'test'

for lang in langs_to_eval:
  print('=== Language', lang)
  for model_size in model_sizes:
    result_path = f'./eval_results/whisper_{model_size}_commonvoice_{lang}_{split}.tsv'
    if os.path.exists(result_path):
      print(f'Results exist already -- skipping:', result_path)
      continue
    model = get_model(model_size, lang)
    batch_size = get_batch_size(model_size)
    print(f'=== Whisper model {model_size}, batch size = {batch_size}')

    ds = get_dataset(dataset_name, split, lang, batch_size)
    results = eval(model, ds, lang)
    save_results(result_path, results)
