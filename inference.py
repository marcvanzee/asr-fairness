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
from langs import COMMONVOICE9_LANGS, VOXPOPULI_LANGS, get_lang_code, ISO_2_TO_3
import dataclasses
import functools


FLAGS = flags.FLAGS

flags.DEFINE_string('db_cache_dir',
                    '/home/marcvanzee/.cache/huggingface/preprocessed_datasets/',
                    'Directory to cache HF datasets after preprocessed')

flags.DEFINE_enum('dataset', 'commonvoice', ['commonvoice', 'voxpopuli'],
                  'Name of dataset to run inference on.')

flags.DEFINE_string('inference_results_dir',
                    './inference_results',
                    'Directory to store inference results tsv files')

flags.DEFINE_enum('model', 'whisper', ['whisper', 'mms'],
                  'Model to run inference on.')

flags.DEFINE_list('whisper_model_sizes',
                  ['tiny', 'base', 'small', 'medium', 'large', 'large-v2'],
                  'Whisper model sizes to run inference on.')

flags.DEFINE_list('mms_model_sizes',
                  ['mms-1b-fl102', 'mms-1b-l1107', 'mms-1b-all'],
                  'MMS model sizes to run inference on.')


# TODO: parallel dataset mapping sometimes gets stuck, investigate why.
_NUM_CORES = 1  # os.cpu_count

# We always run inference on the test split.
_SPLIT = 'test'


@dataclasses.dataclass
class WhisperModel:
  model: whisper.model.Whisper
  options: whisper.decoding.DecodingOptions
  hf_name: str

  @staticmethod
  def preprocess_audio(x):
    return whisper.log_mel_spectrogram(whisper.pad_or_trim(x))
  
  def decode(self, batch):
    outputs = self.model.decode(batch['inputs'].to('cuda'), self.options)
    texts = [output.text for output in outputs]
    return texts


@dataclasses.dataclass
class MMSModel:
  model: Wav2Vec2ForCTC
  processor: AutoProcessor
  hf_name: str

  @staticmethod
  def preprocess_audio(cls, x):
    return cls.processor(x, sampling_rate=16_000, return_tensors='pt')
  
  def decode(self, batch):
    assert len(batch['inputs']) == 1, 'MMS requires a batch size of 1'
    inputs = {k: v.to('cuda') for k, v in batch['inputs'][0].items()}
    with torch.no_grad():
      outputs = self.model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    return [self.processor.decode(ids)]
  

# Note: we store the preprocesss function separately since Arrow datasets cannot
# serialize the Whisper object, so we cannot use an instance of the
# WhisperModel class in preprocessing since the cached dataset will not be
# properly hased.
ModelState = collections.namedtuple('ModelState', 'model, preprocess_fn')


def get_protected_attrs(dataset):
  return {
    'commonvoice': ['age', 'gender', 'accent', 'locale'],
    'voxpopuli': ['gender', 'accent'],
  }[dataset]


def get_batch_size():
  return {
    # Note: we can use larger batch sizes for smaller models, but to avoid
    # too much preprocessing we just use the same batch size everywhere and
    # cache the preprocessed dataset.
    'whisper': 8,
    # The Huggingface interface to MMS only seems to work for single examples.
    'mms': 1,
  }[FLAGS.model]


def get_batches(dataset, lang, audio_preprocess_fn):
  batch_size = get_batch_size()
  def preprocess(hf_name, col_reference, cols_to_keep, cols_to_remove):
    def preprocess_fn(examples):
      inputs = [audio_preprocess_fn(x=x['array'].flatten()) for x in examples['audio']]
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
    
  db_cache_path = os.path.join(FLAGS.db_cache_dir,
                                 f'{dataset}_{lang}_{_SPLIT}_bs_{batch_size}')

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


def get_text_normalizer(lang):
  normalize_fn = {
    'whisper': EnglishTextNormalizer() if lang == 'en' else BasicTextNormalizer(),
    'mms': lambda x: x
  }[FLAGS.model]
  return lambda xs: list(map(normalize_fn, xs))


def save_results(result_path, results):
  keys = list(results.keys())
  print(f'Saving {len(keys[0])} results to {result_path}.')

  with open(result_path, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(keys)
    writer.writerows(zip(*[results[key] for key in keys]))


def get_langs_to_infer(dataset_name, model_size):
  ds_langs = {
    'commonvoice': COMMONVOICE9_LANGS,
    'voxpopuli': VOXPOPULI_LANGS
  }[dataset_name]
  if FLAGS.model == 'whisper':
    whisper_langs = whisper.tokenizer.LANGUAGES
    langs = [x for x in ds_langs if get_lang_code(x) in whisper_langs]
  elif FLAGS.model == 'mms':
    # We create a temporary processor here just to retrieve all languages.
    # The processor loads very fast so it doesn't affect performance much.
    processor = AutoProcessor.from_pretrained('facebook/' + model_size)
    mms_langs_iso3 = list(processor.tokenizer.vocab.keys())
    langs = [lang for lang in ds_langs if ISO_2_TO_3.get(lang) in mms_langs_iso3]
  return langs
  

def get_model_sizes():
  return {
    'whisper': FLAGS.whisper_model_sizes,
    'mms': FLAGS.mms_model_sizes
  }[FLAGS.model]


def load_state(hf_name, lang_code):
  print('=== Loading new model', hf_name)
  if FLAGS.model == 'whisper':
    model = WhisperModel(
      hf_name=hf_name,
      model=whisper.load_model(hf_name),
      options=whisper.DecodingOptions(language=lang_code,
                                      without_timestamps=True))
    preprocessor = WhisperModel.preprocess_audio
  elif FLAGS.model == 'mms':
    model = MMSModel(
      hf_name=hf_name,
      model=Wav2Vec2ForCTC.from_pretrained(hf_name).to('cuda'),
      processor=AutoProcessor.from_pretrained(hf_name)
    )
    preprocessor = functools.partial(MMSModel.preprocess_audio, cls=model)

  return ModelState(model=model, preprocess_fn=preprocessor)


def update_model_state(state, model_size, lang):
  model = state.model
  lang_code = get_lang_code(lang)
  use_en_model = lang == 'en' and 'large' not in model_size
  hf_name = {
    'mms': 'facebook/' + model_size,
    'whisper': model_size + '.en' if use_en_model else model_size
  }[FLAGS.model]

  if not model or model.hf_name != hf_name:
    state = load_state(hf_name, lang_code)
  elif FLAGS.model == 'whisper':
    print('Updating Whisper decoder with new language', lang_code)
    # Whisper is multilingual or en-only, which is indicated by hf_name. If
    # we switch language we only have to update the decoding options.
    state.model.options = whisper.DecodingOptions(
      language=lang_code, without_timestamps=True)
  elif FLAGS.model == 'mms':
    print('Updating MMS tokenizer and adapter with new language', lang_code)
    # Keep the same model in memory and simply switch out the language.
    lang_alpha3 = ISO_2_TO_3.get(lang_code)
    assert lang_alpha3, f'Invalid language for MMS: {lang_alpha3}'
    model.processor.tokenizer.set_target_lang(lang_alpha3)
    model.model.load_adapter(lang_alpha3)

  return state


def main(_):
  model_name = FLAGS.model
  model_sizes = get_model_sizes()
  dataset = FLAGS.dataset

  print( ' -------------- Inference settings --------------------------------')
  print( '| Model', FLAGS.model)
  print(f'| Models sizes: {model_sizes}')
  print( '| Dataset', dataset)
  print( '|-------------------------------------------------------------------')

  state = ModelState(model=None, preprocess_fn=None)

  # First iterate over model size so we can keep the same model in memory for
  # all languages -- dataset loading is quite fast once it is cached.
  for model_size in model_sizes:
    print('=== Running inference on model size', model_size)
    langs_to_infer = get_langs_to_infer(dataset, model_size)
    print(f'=== Running inference on {len(langs_to_infer)} languages: {langs_to_infer}') 
    for lang in langs_to_infer:
      print('=== Running inference on language', lang)
      result_path = os.path.join(
        FLAGS.inference_results_dir,
        f'{model_name}_{model_size}_{FLAGS.dataset}_{lang}_{_SPLIT}.tsv')
      if os.path.exists(result_path):
        print(f'Results exist already -- skipping:', result_path)
        continue

      state = update_model_state(state, model_size, lang)
      results = collections.defaultdict(list)
      for batch in tqdm(get_batches(dataset, lang, state.preprocess_fn)):
        texts = state.model.decode(batch)
        results['reference_original'].extend(batch['reference_original'])
        results['inferred_original'].extend(texts)
        for key in get_protected_attrs(dataset):
          results[key].extend(batch[key])

      normalize = get_text_normalizer(lang)
      results['reference'] = normalize(results['reference_original'])
      results['inferred'] = normalize(results['inferred_original'])

      save_results(result_path, results)


if __name__ == '__main__':
  app.run(main)
