## Performance of ASR models on ES and PT Latin America


import json
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

from absl import app
from collections import OrderedDict
from scipy import stats
from tabulate import tabulate
from generate_eval_data_es import EvalInfo
from langs import get_lang_code, COMMONVOICE9_LANGS, VOXPOPULI_LANGS
import os
import collections
import dataclasses
from typing import List

# Is there performance disparity for accents - independent of age and gender?
# Do these disparities change if we add demographic information to the mix?	
ES_ACCENTS = ["Iberian", "Mexican", "CentralAmerican", "Chilean", "RiverPlatean", "Andean", "Caribbean"]
OUTPUT_DIR = './visualizations/'

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

def get_available_models(models):
  all_models = [
    'whisper_tiny', 'whisper_base', 'whisper_small', 'whisper_medium',
    'whisper_large', 'whisper_large-v2', 'mms_mms-1b-fl102', 'mms_mms-1b-l1107',
    'mms_mms-1b-all']
  available_models = [x for x in all_models if x in models]
  if len(available_models) < len(all_models):
    print(f'MISSING MODELS FOR WER TABLE!', set(all_models)-set(models))
  return available_models


def read_results():
  with open('results/eval_data_es.json') as f:
    return json.load(f)


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

def do_significance_testing(demo_group_wers, others_wers, demo, other):
	# demo_group_wers is test group
	# other_wers is the group to test against (all, or specific other group)

	test_result = stats.ttest_ind(demo_group_wers, others_wers)

	if test_result[1] < 0.05: 
		# negative t-value means that A has the lowest mean. B is negatively disparate.
		if test_result[0] < 0:
			return other, test_result[1]
		elif test_result[0] > 0:
			return demo, test_result[1]
	else:
		return None, None


def make_heatmap(results):
  print('Heatmap of three-way intersectionality -- gender, age and accent -- across model size.\n')

  models_results = []
  models_results_groups = []
  models = [x for x in get_available_models(results.keys()) if "whisper" in x]
  for model in models:

    #if not "whisper" in model: continue
    model_results = [0] * len(ES_ACCENTS) # rows in table
    model_results_groups = [''] * len(ES_ACCENTS) # rows in table
    all_demographic_groups = [x for x in results[model].keys() if len(x.split("_")) == 3 and "other" not in x and "US" not in x]

    for demo_group in all_demographic_groups:
      demo_group_wers = results[model][demo_group].wers
      demo_group_accent = demo_group.split("_")[-1]
      demo_group_gender_age = "_".join(demo_group.split("_")[:-1])

      all_other_demos = [x for demo, x in results[model].items() if demo != demo_group]
      all_other_wers =  [y for x in all_other_demos for y in x.wers]
      sig, p = do_significance_testing(demo_group_wers, all_other_wers, demo_group_gender_age, "all")
      if sig:
        if sig == "all":
          model_results[ES_ACCENTS.index(demo_group_accent)] += -1 # "all"
        else:
          model_results[ES_ACCENTS.index(demo_group_accent)] += 1 # demo_group_gender_age

        model_results_groups[ES_ACCENTS.index(demo_group_accent)] += f" {sig} "

    models_results.append(model_results)
    models_results_groups.append(model_results_groups)

  print("Table of three-way intersectionality -- gender, age and accent -- across model size.")
  table = [["Model"]+ES_ACCENTS]
  for e,m in enumerate(models):
  	table.append([m]+models_results_groups[e])
  print(tabulate(table, headers='firstrow', tablefmt='latex'))

  models_results = np.array(models_results)
  fig, ax = plt.subplots()
  im = ax.imshow(models_results)

  # Show all ticks and label them with the respective list entries
  ax.set_xticks(np.arange(len(ES_ACCENTS)), labels=ES_ACCENTS)
  ax.set_yticks(np.arange(len(models)), labels=models)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for i in range(len(models)):
    for j in range(len(ES_ACCENTS)):
      text = ax.text(j, i, models_results[i, j], ha='center', va='center', color='w')

  ax.set_title('Intersectionality disparity across models')
  fig.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, 'intersectionality_disparity_es.png'))
  print()


def write_wer_table(results): 
	print("WER table for accents of Spanish across model size")

	table = [['model']+ES_ACCENTS]
	models = [x for x in get_available_models(results.keys()) if "whisper" in x]
  
	for model in models:
		wer_info = results[model]
		latinamerican = []
		for accent in ES_ACCENTS:
			#if accent != "Iberian":
			try:
				other_demographic_groups = [demo for demo in wer_info.keys() if "_" not in demo and demo != accent and demo != "All"] # no intersectional
				wer_info_other_demographic_groups = [wer_info[demo].wers for demo in other_demographic_groups]
				all_other_wers =  [y for x in wer_info_other_demographic_groups for y in x]
				sig = do_significance_testing(wer_info[accent].wers, all_other_wers, accent, "all")
				
				if sig[0]:
					if sig[1] < 0.001: 
						star = "***"

					elif 0.001 < sig[1] < 0.01: 
						star = "**"

					elif 0.01 < sig[1] < 0.05: 
						star = "*"

					latinamerican.append(star)
				else: latinamerican.append(None)
			except ValueError:
				print("No value")
		out = [model] + latinamerican # , wer_info["Iberian"].wer]
		
		table.append(out)
	print(tabulate(table, headers='firstrow', tablefmt='latex'))
	print()

def write_wer_table_gender(results): 
	print("WER table for accents/gender of Spanish across model size")
	
	table = [['model'] + ES_ACCENTS]
	models = [x for x in get_available_models(results.keys()) if "whisper" in x]
	for model in models:
		wer_info = results[model]
		wers_table_info = {"female": [], "male": []}
		accents = []
		for gender in ["female", "male"]:
			wer_results = []
			for accent in ES_ACCENTS:
				demo_group = f"{gender}_{accent}"
				try:
					other_demographic_groups = [demo for demo in wer_info.keys() if len(demo.split("_")) == 2 and "other" not in demo and "US" not in demo and demo != demo_group]
					wer_info_other_demographic_groups = [wer_info[demo].wers for demo in other_demographic_groups]
					all_other_wers =  [y for x in wer_info_other_demographic_groups for y in x]
					sig = do_significance_testing(wer_info[demo_group].wers, all_other_wers, demo_group, "all")
					wers_table_info[gender].append(wer_info[demo_group].wer)
					accents.append(accent)
						
					if sig[0]:

						if sig[1] < 0.001: 
							star = "***"

						elif 0.001 < sig[1] < 0.01: 
							star = "**"

						elif 0.01 < sig[1] < 0.05: 
							star = "*"

						wer_results.append(star)
					else: wer_results.append(None)
				except ValueError:
					print("No value")
			out = [f"{model}_{gender}"] + wer_results
		
			table.append(out)
		print(len(wers_table_info["female"]))
		print(len(accents))
		for i in range(int(len(accents)/2)):
			print(i)
			print(accents[i])
			print("FEMALE:\t", wers_table_info["female"][i])
			print("MALE:\t", wers_table_info["male"][i])
	print(tabulate(table, headers='firstrow', tablefmt='latex'))
	print()

def plot_wer_by_model_size(results):
	print('Plot of WER for accents in Spanish across model size')

	models = [x for x in get_available_models(results.keys()) if "whisper" in x]
	
	plt.rcParams['axes.prop_cycle'] = cycler(color=['red',"pink",'orange','yellow',"violet",'green','blue','purple'])
	plt.clf() 
	plt.figure(figsize = (8,4))
	plt.title(f'Performance (WER) by Whisper family models')
	
	sizes_and_wers_to_plot = {"All": []}
	markerson = {"All": []}

	for accent in ES_ACCENTS:
		sizes_and_wers_to_plot[f"{accent}"] = []
		markerson[f"{accent}"] = []

	for i, model in enumerate(models):
		for accent in ES_ACCENTS: 
			wer_info = results[model].get(f"{accent}", None)
			all_other_wers = [results[model].get(f"{accent1}", None) for accent1 in ES_ACCENTS if accent1 != accent]
			all_other_wers = [y for x in all_other_wers for y in x.wers]
			sig = stats.ttest_ind(wer_info.wers, all_other_wers).pvalue
			if sig < 0.05:
				print(sig)
				markerson[accent].append(i)
			sizes_and_wers_to_plot[accent].append((
				MODEL_PARAMETERS[model],
				wer_info.wer))
		sizes_and_wers_to_plot["All"].append((
				MODEL_PARAMETERS[model],
				results[model].get(f"All", None).wer))

	for accent in ES_ACCENTS+["All"]:
		plt.plot(*zip(*sizes_and_wers_to_plot[accent]), marker='o', label=f'{accent}', markevery=markerson[accent])
	plt.xlabel('Model parameters')
	plt.ylabel('Performance (WER)')
	plt.xscale("log")
	plt.legend(loc='upper right', fontsize='8')
	plt.savefig(os.path.join(OUTPUT_DIR, f'wer_by_model_size_es.png'))
	

def plot_wer_by_model_size_rel(results):
	print('Plot of relative WER for m/f in all dialects of Spanish across model size')

	models = [x for x in get_available_models(results.keys()) if "whisper" in x]
	
	plt.rcParams['axes.prop_cycle'] = cycler(color=['red',"pink",'orange','yellow',"violet",'green','blue','purple'])
	plt.clf() 
	plt.figure(figsize = (8,4))
	#plt.suptitle(f'Performance (WER) and Disparity on Spanish')
	#plt.subplot(1, 2, 1) # row 1, col 2 index 1 --> absolute WERS
	plt.title(f'Disparity') #Performance (WER)')
	
	# sizes_and_wers_to_plot = {}

	# for accent in ES_ACCENTS:
	# 	sizes_and_wers_to_plot[f"{accent}"] = []

	# for model in models:
	# 	for accent in ES_ACCENTS: 
	# 		wer_info = results[model].get(f"{accent}", None)
	# 		sizes_and_wers_to_plot[accent].append((
	# 			MODEL_PARAMETERS[model],
	# 			wer_info.wer))

	# for accent in ES_ACCENTS:
	# 	plt.plot(*zip(*sizes_and_wers_to_plot[accent]), label=f'{accent}')
	# plt.xlabel('Model parameters')
	# plt.ylabel('Performance (WER)')
	# plt.xscale("log")
	# plt.legend(loc='upper right', fontsize='8')

	# plt.subplot(1, 2, 2) # row 1, col 2 index 1 --> relative WERS

	# plt.title(f'Disparity') # female-male
	# plt.axhline(y = 0, color = 'r', linestyle = '--')

	# #sizes_and_wers_to_plot = []
	sizes_and_wers_to_plot = {"All": []}
	markerson = {"All": []}

	for accent in ES_ACCENTS:
		sizes_and_wers_to_plot[f"{accent}"] = []
		markerson[f"{accent}"] = []

	for i, model in enumerate(models):
		for accent in ES_ACCENTS: 
			wer_info_f = results[model].get(f"female_{accent}", None)
			wer_info_m = results[model].get(f"male_{accent}", None)
			if not (f:=wer_info_f) or not (m:=wer_info_m):
				print(f'{model=} missing gender {"male" if f else "male"}')
				continue
			sig = stats.ttest_ind(wer_info_f.wers, wer_info_m.wers).pvalue
			if sig < 0.05:
				markerson[accent].append(i)
			sizes_and_wers_to_plot[accent].append((
				MODEL_PARAMETERS[model],
				rel_wer_diff(wer_info_f.wer, wer_info_m.wer)))
		sizes_and_wers_to_plot["All"].append((
				MODEL_PARAMETERS[model],
				rel_wer_diff(results[model].get(f"female", None).wer, results[model].get(f"male", None).wer)))

	for accent in ES_ACCENTS+["All"]:
		plt.plot(*zip(*sizes_and_wers_to_plot[accent]), marker='o', label=f'{accent}', markevery=markerson[accent]) #very
	plt.xlabel('Model parameters')
	plt.ylabel('Disparity (f-m)')
	plt.xscale("log")
	plt.legend(loc='upper right', fontsize='8')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, f'wer_by_model_size_es_rel.png'))


def get_wer_analysis():

  results = read_results()
  
  for model in results.keys():
    outputs[model] = collections.defaultdict(WerInfo)
    for demographic_group, eval_info_dict in results[model].items():
      for x in eval_info_dict:
        eval_info = EvalInfo(*x)
        outputs[model][demographic_group].add_eval_result(eval_info)

outputs = {}

def main(_):
  get_wer_analysis()
  #write_wer_table(outputs)
  write_wer_table_gender(outputs)
  #make_heatmap(outputs)
  #plot_wer_by_model_size(outputs)
  #plot_wer_by_model_size_rel(outputs)

if __name__ == '__main__':
  app.run(main)

