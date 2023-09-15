import json
import matplotlib.pyplot as plt
import numpy as np

from absl import app
from collections import OrderedDict
from scipy import stats
from tabulate import tabulate
from generate_eval_data import EvalInfo
from langs import get_lang_code, COMMONVOICE9_LANGS, VOXPOPULI_LANGS
import os
import collections
import dataclasses
from typing import List


MODEL_PARAMETERS = OrderedDict({
    'whisper_tiny': 39000000,
    'whisper_base': 74000000,
    'whisper_small': 244000000,
    'whisper_medium': 769000000,
    'whisper_large': 1550000000,
    'whisper_large-v2': 1550000000,
    'mms_mms-1b-fl102': 1000000000,
    'mms_mms-1b-l1107': 1000000000,
    'mms_mms-1b-all': 1000000000,
}) 

OUTPUT_DIR = './visualizations/'


def make_plot_genderdiff(gender_significant_results, total_runs):
  # Plots modelsize (x)/disparaty rate (y)
  models = [m for m in MODEL_PARAMETERS.keys() if "whisper" in m]
  x = [p for m, p in MODEL_PARAMETERS.items() if "whisper" in m]
  y = [100*((gender_significant_results[model]['female']+gender_significant_results[model]['male'])/total_runs[model]) for model in models]

  plt.plot(x, y, 'go-')  
  plt.xlabel('Model size')
  plt.ylabel('Disparity rate (%)')
  plt.xticks(x, models, rotation=45)
  plt.subplots_adjust(bottom=0.3)
  plt.savefig(os.path.join(OUTPUT_DIR, 'gender_disparity.png'))


def visualize_per_model_significance(gender_significant_results, total_runs):
  table = [['Model', '\\female', '\\male', 'all sig.', 'datasets', 'per of datasets']]
  total = ['Total', 0, 0, 0, 0]
  for model in list(MODEL_PARAMETERS.keys()): #sorted(gender_significant_results.keys()):
    model_runs = total_runs[model]
    model_info = [model, gender_significant_results[model]['female'], gender_significant_results[model]['male'], gender_significant_results[model]['female']+gender_significant_results[model]['male'], round(100*(gender_significant_results[model]['female']+gender_significant_results[model]['male'])/model_runs,1)]
    table.append(model_info)

    total[1] += model_info[1] 
    total[2] += model_info[2] 
    total[3] += model_info[1] + model_info[2] 
  total[4] = round(100*(total[3]/sum(total_runs.values())),1)
  table.append(total)

  print(tabulate(table, headers='firstrow', tablefmt='latex'))


def make_heatmap(intersectionality_significant_results, total_runs):
  
  models = list(MODEL_PARAMETERS.keys())
  models_results = []
  intersectionality_groups = ['female_teens', 'male_teens', 'other_teens', 'female_twenties', 'male_twenties', 'other_twenties', 'female_thirties', 'male_thirties', 'other_thirties', 'female_fourties', 'male_fourties', 'other_fourties', 'female_fifties', 'male_fifties', 'female_sixties', 'male_sixties', 'male_seventies'] #, 'female_eighties', 'male_nineties']

  # for model in models:
  #   for demographic_group in intersectionality_significant_results[model]:
  #     intersectionality_groups.add(demographic_group)
  # print(intersectionality_groups)

  for model in models:
    model_results = []
    for intersectionality_group in intersectionality_groups:
      if intersectionality_group in intersectionality_significant_results[model]:
        model_results.append(intersectionality_significant_results[model][intersectionality_group])
      else:
        model_results.append(0)
    models_results.append(model_results)  

  models_results = np.array(models_results)
  fig, ax = plt.subplots()
  im = ax.imshow(models_results)

  # Show all ticks and label them with the respective list entries
  ax.set_xticks(np.arange(len(intersectionality_groups)), labels=intersectionality_groups)
  ax.set_yticks(np.arange(len(models)), labels=models)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for i in range(len(models)):
      for j in range(len(intersectionality_groups)):
          text = ax.text(j, i, models_results[i, j],
                         ha='center', va='center', color='w')

  ax.set_title('Intersectionality disparity per model')
  fig.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, 'intersectionality_disparity.png'))


def read_results():
  with open('results/eval_data.json') as f:
    return json.load(f)


def do_significance_testing(wers, x_demo, z_demo):
  
  # intersectionality
  if z_demo == 'all':
    
    other_demographic_groups = [x for x in wers.keys() if x != 'female' and x != 'male' and x != x_demo]
    wers_other_demographic_groups = [wers[x] for x in other_demographic_groups]
    all_other_wers = [item for row in wers_other_demographic_groups for item in row]
    
    test_result = stats.ttest_ind(wers[x_demo], all_other_wers)

  # gender
  else:
    test_result = stats.ttest_ind(wers[x_demo], wers[z_demo])
  
  if test_result[1] < 0.05:    
    # negative t-value means that x_demo has the lowest mean. Z-demo is negatively disparate.
    if test_result[0] < 0:
      return z_demo
    elif test_result[0] > 0:
      return x_demo
  else:
    return None


def get_significance(results, gender_disparity=True, intersectionality_disparity=True):

  gender_significant_results = {}
  intersectionality_significant_results = {} # counter per model
  total_runs = {}

  for dataset in results: # cv vp
    for model in results[dataset]: # tiny, small, mms-fl102
      
      if model not in total_runs:
        total_runs[model] = 0
        
        if model not in gender_significant_results.keys():
          gender_significant_results[model] = {'female': 0, 'male': 0}
        if model not in intersectionality_significant_results.keys():
          intersectionality_significant_results[model] = {}
      
      for language in results[dataset][model]: # nl, fr
        total_runs[model] += 1

        try:
          wers = {}
          dem_group_to_eval_data = results[dataset][model][language]
          for demographic_group, wer_res in dem_group_to_eval_data.items():
            # if len(wer_res) < 5:
            #   print(language, demographic_group, wer_res)
            wers[demographic_group] = [EvalInfo(*x).wer for x in wer_res]

          if gender_disparity:

            gender_sign = do_significance_testing(wers, 'female', 'male')
            if gender_sign:
              gender_significant_results[model][f'{gender_sign}'] += 1

          if intersectionality_disparity:
            if dataset == 'voxpopuli': continue

            # check that both genders have age distributions:
            string_demographic_groups = '+'.join(wers.keys())
            if not 'female_' in string_demographic_groups and 'male_' in string_demographic_groups: continue
            
            intersectionality_groups = [x for x in wers.keys() if '_' in x]
            for demographic_group in intersectionality_groups:
              intersec_sign = do_significance_testing(wers, demographic_group, 'all')
                
              if intersec_sign:
                #if intersec_sign == 'all': continue
                if demographic_group not in intersectionality_significant_results[model].keys():
                  intersectionality_significant_results[model][f'{demographic_group}'] = 0
                intersectionality_significant_results[model][f'{demographic_group}'] += 1
         
        except KeyError: continue

  f_sig = sum([gender_significant_results[model]['female'] for model in gender_significant_results.keys()])
  m_sig = sum([gender_significant_results[model]['male'] for model in gender_significant_results.keys()])
  total = sum(total_runs.values())

  print(f"There are {total} experiments. Of these {(f_sig+m_sig)/total} % are significant. {f_sig/total} are disparate against women, {m_sig/total} are disparate against men.")

  if gender_disparity:
    make_plot_genderdiff(gender_significant_results, total_runs)
    #visualize_per_model_significance(gender_significant_results, total_runs) 

  if intersectionality_disparity:
    make_heatmap(intersectionality_significant_results, total_runs)


@dataclasses.dataclass
class WerInfo:
  total_ref_len: int = 0
  total_updates: int = 0
  wers: List[float] = dataclasses.field(default_factory=list)
  
  def add_eval_result(self, eval_info: EvalInfo):
    self.total_updates += eval_info.wer * eval_info.ref_len
    self.total_ref_len += eval_info.ref_len
    self.wers.append(eval_info.wer)

  @property
  def wer(self):
    return self.total_updates / self.total_ref_len


def rel_wer_diff(female_wer: float, male_wer: float):
  return (female_wer - male_wer) / max(female_wer, male_wer)


def get_available_models(models):
  all_models = [
    'whisper_tiny', 'whisper_base', 'whisper_small', 'whisper_medium',
    'whisper_large', 'whisper_large-v2', 'mms_mms-1b-fl102', 'mms_mms-1b-l1107',
    'mms_mms-1b-all']
  available_models = [x for x in all_models if x in models]
  if len(available_models) < len(all_models):
    print(f'MISSING MODELS FOR WER TABLE!', set(all_models)-set(models))
  return available_models


def write_wer_table(results, dataset):
  table = [['model', 'wer_female', 'wer_male', 'wer_diff', 'rel_wer_diff']]
  results = results[dataset]

  for model in get_available_models(results.keys()):
    wer_info_by_group = results[model]['all']
    if any(x not in wer_info_by_group for x in ['male', 'female']):
      print(f'MISSING GENDER IN WER TABLE {model=}')
      continue
    wer_info_female = wer_info_by_group['female']
    wer_info_male = wer_info_by_group['male']
    wer_female = wer_info_female.wer
    wer_male = wer_info_male.wer
    wer_diff = wer_female - wer_male
    rel_diff = rel_wer_diff(wer_female, wer_male)
    table.append([
      model, wer_info_male.wer, wer_info_female.wer, wer_diff, rel_diff])
  
  print(tabulate(table, headers='firstrow', tablefmt='latex'))


def plot_wer_by_year(results, model_filter):
  # Plots time (x)/wer rate (y) per model
  models = get_available_models(results.keys())

  def filter_fn(model):
    if not model_filter: return False
    if isinstance(model_filter, str) and model_filter not in model: return True
    if isinstance(model_filter, list) and model not in model_filter: return True
    return False

  years = list(range(2009, 2018))
  years_labels = [years[i] for i in range(len(years)) if (i+1) % 2 == 0]

  plt.clf()
  plt.figure(figsize = (8,4))
  plt.suptitle(f'WER by year on Voxpopuli')

  plt.subplot(1, 2, 1) # row 1, col 2 index 1 --> absolute WERS

  plt.title(f'Absolute WER')

  for model in models:
    if filter_fn(model):
      continue
    print(model)
    years_wers_to_plot_m = []
    years_wers_to_plot_f = []
    for year in years:
      wer_info_f = results[model]['all'].get(f'female_{year}', None)
      wer_info_m = results[model]['all'].get(f'male_{year}', None)
      if wer_info_f:
        years_wers_to_plot_f.append((year, wer_info_f.wer))
      if wer_info_m:
        years_wers_to_plot_m.append((year, wer_info_m.wer))
    p, = plt.plot(*zip(*years_wers_to_plot_f), label=model+'_female')
    plt.plot(*zip(*years_wers_to_plot_m), label=model+'_male',
             color=p.get_color(), linestyle='--')

  plt.xlabel('Year')
  plt.xticks(years_labels, rotation=45)
  plt.ylabel(f'absolute WER')
  plt.legend(loc='upper right', fontsize='8')

  plt.subplot(1, 2, 2) # index 2 --> relative WERS
  
  plt.title(f'Relative WER female-male')
  plt.axhline(y = 0, color = 'k', linestyle = '--')

  for model in models:
    if filter_fn(model):
      continue
    years_wers_to_plot = []
    markerson = []
    for i, year in enumerate(years):
      wer_info_f = results[model]['all'].get(f'female_{year}', None)
      wer_info_m = results[model]['all'].get(f'male_{year}', None)
      if not (f:=wer_info_f) or not (m:=wer_info_m):
        print(f'{model=}, {year=} missing gender {"male" if f else "male"}')
        continue
      if stats.ttest_ind(wer_info_f.wers, wer_info_m.wers).pvalue < 0.05:
        markerson.append(i)
      rel_wer_d = rel_wer_diff(wer_info_f.wer, wer_info_m.wer)
      years_wers_to_plot.append((year, rel_wer_d))
    plt.plot(*zip(*years_wers_to_plot), label=model, marker='o',
             markevery=markerson)
  
  plt.xlabel('Year')
  plt.xticks(years_labels, rotation=45)
  plt.ylabel('relative WER (f-m)')
  plt.legend(loc='upper right', fontsize='8')
  plt.tight_layout()
  filename = f'wer_by_year_{model_filter}.png'
  plt.savefig(os.path.join(OUTPUT_DIR, filename))
  print('Wrote to', filename)


def plot_wer_by_model_size(results, dataset):
  results = results[dataset]
  models = [x for x in get_available_models(results.keys()) if 'whisper' in x]

  plt.clf() 
  plt.figure(figsize = (8,4))
  plt.suptitle(f'Performance (WER) by Whisper family model size')
  plt.subplot(1, 2, 1) # row 1, col 2 index 1 --> absolute WERS

  plt.title(f'Performance (WER)')

  sizes_and_wers_m_to_plot = []
  sizes_and_wers_f_to_plot = []

  for model in models:
    wer_info_f = results[model]['all'].get(f'female', None)
    if wer_info_f:
      sizes_and_wers_f_to_plot.append((
        MODEL_PARAMETERS[model],
        wer_info_f.wer))
    wer_info_m = results[model]['all'].get(f'male', None)
    if wer_info_m:
      sizes_and_wers_m_to_plot.append((
        MODEL_PARAMETERS[model],
        wer_info_m.wer))
    
  p, = plt.plot(*zip(*sizes_and_wers_f_to_plot), label='female')
  plt.plot(*zip(*sizes_and_wers_m_to_plot), label='male',
           color=p.get_color(), linestyle='--')
  plt.xlabel('Model parameters')
  plt.ylabel('Performance (WER)')
  plt.xscale("log")
  plt.legend(loc='upper right', fontsize='8')

  plt.subplot(1, 2, 2) # row 1, col 2 index 1 --> relative WERS

  plt.title(f'Disparity') # female-male
  plt.axhline(y = 0, color = 'r', linestyle = '--')

  sizes_and_wers_to_plot = []
  markerson = []
  for i, model in enumerate(models):
    wer_info_f = results[model]['all'].get(f'female', None)
    wer_info_m = results[model]['all'].get(f'male', None)
    if not (f:=wer_info_f) or not (m:=wer_info_m):
      print(f'{model=} missing gender {"male" if f else "male"}')
      continue
    if stats.ttest_ind(wer_info_f.wers, wer_info_m.wers).pvalue < 0.05:
      markerson.append(i)

    sizes_and_wers_to_plot.append((
      MODEL_PARAMETERS[model],
      rel_wer_diff(wer_info_f.wer, wer_info_m.wer)))
  
  plt.plot(*zip(*sizes_and_wers_to_plot), marker='o', label='wer',
           markevery=markerson)
  plt.xlabel('Model parameters')
  plt.ylabel('Disparity (f-m)')
  plt.xscale("log")
  plt.legend(loc='upper right', fontsize='8')
  plt.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, f'wer_by_model_size_{dataset}.png'))


def plot_wer_by_model_size_per_lang(results, dataset, langs_to_show=None, name=None):
  results = results[dataset]
  models = [x for x in get_available_models(results.keys()) if 'whisper' in x]

  plt.clf()
  plt.title(f'{dataset} relative WER by model size (log scale)')
  plt.axhline(y = 0, color = 'r', linestyle = '--')

  if langs_to_show:
    langs = langs_to_show
  else:
    langs = list(list(results.values())[0].keys())  # ugly.

  for lang in langs:
    model_sizes_to_plot = []
    wers_to_plot = []
    for model in models:
      lang_dict = results[model].get(lang, None)
      if not lang_dict:
        print('MISSING language', lang, 'for model', model, '--- skipping')
        continue
      wer_info_f = lang_dict.get(f'female', None)
      wer_info_m = lang_dict.get(f'male', None)
      if not (f:=wer_info_f) or not (m:=wer_info_m):
        print(f'{model=} {lang=} missing gender {"male" if f else "male"}')
        continue
      model_sizes_to_plot.append(MODEL_PARAMETERS[model])
      wers_to_plot.append(rel_wer_diff(wer_info_f.wer, wer_info_m.wer))
    
    plt.plot(model_sizes_to_plot, wers_to_plot, label=lang)
    plt.xlabel('Model parameters')
    plt.ylabel('relative WER (f-m)')
    plt.xscale("log")
    plt.legend(loc='upper right', fontsize='8')

  fig_name = f'wer_by_model_size_{dataset}_per_lang.png'
  if langs_to_show:
    fig_name = f'wer_by_model_size_{dataset}_per_lang_{name}.png'

  plt.savefig(os.path.join(OUTPUT_DIR, fig_name))


def get_wer_analysis(results):
  outputs = {
    'commonvoice': collections.defaultdict(dict),
    'voxpopuli': collections.defaultdict(dict),
    'all': collections.defaultdict(dict),
  }

  def _add_to_all_output(model, lang, gender, eval_info):
    def _add(lang):
      if lang not in outputs['all'][model]:
        outputs['all'][model][lang] = collections.defaultdict(WerInfo)
      outputs['all'][model][lang][gender].add_eval_result(eval_info)
    _add(lang)
    _add('all')

  for dataset, model_dict in results.items():
    for model, lang_dict in model_dict.items():
      all_wer_info = collections.defaultdict(WerInfo)
      for lang, eval_info_dict in lang_dict.items():
        wer_info_by_gender = collections.defaultdict(WerInfo)
        for gender in ['male', 'female']:
          if gender not in eval_info_dict:
            continue
          for x in eval_info_dict[gender]:
            eval_info = EvalInfo(*x)
            all_wer_info[gender].add_eval_result(eval_info)
            wer_info_by_gender[gender].add_eval_result(eval_info)
            _add_to_all_output(model, lang, gender, eval_info)
            if dataset == 'voxpopuli':
              year = eval_info.date[:4]
              all_wer_info[f'{gender}_{year}'].add_eval_result(eval_info)
              wer_info_by_gender[f'{gender}_{year}'].add_eval_result(eval_info)
              _add_to_all_output(model, lang, gender, eval_info)
        outputs[dataset][model][lang] = wer_info_by_gender
      
      outputs[dataset][model]['all'] = all_wer_info

  for dataset in ['commonvoice', 'voxpopuli']:
    print('------', dataset)
    write_wer_table(outputs, dataset)
    plot_wer_by_model_size(outputs, dataset)

    plot_wer_by_model_size_per_lang(outputs, dataset)
    # plot_wer_by_model_size_per_lang(outputs, dataset, small_langs, 'small_langs')
    # plot_wer_by_model_size_per_lang(outputs, dataset, big_langs, 'big_langs')

  plot_wer_by_model_size(outputs, 'all')

  # One graph for all Whisper models and one for all MMS models.
  for model in ['whisper', 'mms']:
    plot_wer_by_year(outputs['voxpopuli'], model)

  for models_to_compare in [
    ['whisper_large-v2', 'mms_mms-1b-all'],
    ['whisper_tiny', 'whisper_large-v2'],
    ['mms_mms-1b-fl102', 'mms_mms-1b-all']]:
    plot_wer_by_year(outputs['voxpopuli'], models_to_compare)

  plot_wer_by_year(outputs['voxpopuli'], 'whisper')


def main(_):
  results = read_results()
  get_significance(results, gender_disparity=True, intersectionality_disparity=True)
  #get_wer_analysis(results)


if __name__ == '__main__':
  app.run(main)



