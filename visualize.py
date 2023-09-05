import json
import matplotlib.pyplot as plt
import numpy as np

from absl import app
from collections import OrderedDict
from scipy import stats
from tabulate import tabulate

MODEL_PARAMETERS = OrderedDict({
    "whisper_tiny": 39000000,
    "whisper_base": 74000000,
    "whisper_small": 244000000,
    "whisper_medium": 769000000,
    "whisper_large": 1550000000}
) # "mms_mms-1b-all": 1000000000, (between whipser_medium and whisper_large)

OUTPUT_DIR = './visualizations/'


def make_plot_genderdiff(gender_significant_results, total_runs):
  # Plots modelsize (x)/disparaty rate (y)
  models = list(MODEL_PARAMETERS.keys())
  x = list(MODEL_PARAMETERS.values())
  y = [100*((gender_significant_results[model]["female"]+gender_significant_results[model]["male"])/total_runs[model]) for model in models]
  
  print("x", x)
  print("y", y)
  print("labels", models)

  plt.plot(x, y, 'go-')  
  plt.xlabel('Model size')
  plt.ylabel('Disparity rate (%)')
  plt.xticks(x, models, rotation=45)
  plt.subplots_adjust(bottom=0.3)
  plt.savefig(os.path.join(OUTPUT_DIR, "gender_disparity.png"))


def visualize_per_model_significance(gender_significant_results, total_runs):
  table = [["Model", "\\female", "\\male", "all sig.", "datasets", "per of datasets"]]
  total = ["Total", 0, 0, 0, 0]
  for model in list(MODEL_PARAMETERS.keys()): #sorted(gender_significant_results.keys()):
    model_runs = total_runs[model]
    model_info = [model, gender_significant_results[model]['female'], gender_significant_results[model]['male'], gender_significant_results[model]['female']+gender_significant_results[model]['male'], round(100*(gender_significant_results[model]['female']+gender_significant_results[model]['male'])/model_runs,1)]
    table.append(model_info)

    total[1] += model_info[1] 
    total[2] += model_info[2] 
    total[3] += model_info[1] + model_info[2] 
  total[4] = round(100*(total[3]/sum(total_runs.values())),1)
  table.append(total)

  print(tabulate(table, headers="firstrow", tablefmt="latex"))


def make_heatmap(intersectionality_significant_results, total_runs):
  
  models = list(MODEL_PARAMETERS.keys())
  models_results = []
  intersectionality_groups = ['female_teens', 'male_teens', 'other_teens', 'female_twenties', 'male_twenties', 'other_twenties', 'female_thirties', 'male_thirties', 'other_thirties', 'female_fourties', 'male_fourties', 'other_fourties', 'female_fifties', 'male_fifties', 'female_sixties', 'male_sixties', 'male_seventies', 'female_eighties', 'male_nineties']

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
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  for i in range(len(models)):
      for j in range(len(intersectionality_groups)):
          text = ax.text(j, i, models_results[i, j],
                         ha="center", va="center", color="w")

  ax.set_title("Intersectionality disparity per model")
  fig.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, "intersectionality_disparity.png"))


def read_results():
  with open('results/wers.json', 'r') as f:
    return json.load(f)


def do_significance_testing(wers, x_demo, z_demo):

  # intersectionality
  if z_demo == "all":
    other_demographic_groups = [x for x in wers.keys() if x != "female" and x != "male" and x != x_demo]
    wers_other_demographic_groups = [wers[x] for x in other_demographic_groups]
    all_other_wers = [item for row in wers_other_demographic_groups for item in row]
    test_result = stats.ttest_ind(wers[x_demo], all_other_wers)
  
  # gender
  else:
    test_result = stats.ttest_ind(wers[x_demo], wers[z_demo])
  
  if test_result[1] < 0.05:
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
          gender_significant_results[model] = {"female": 0, "male": 0}
        if model not in intersectionality_significant_results.keys():
          intersectionality_significant_results[model] = {}
      
      for language in results[dataset][model]: # nl, fr
        total_runs[model] += 1

        try:
          wers = {}
          for demographic_group in results[dataset][model][language].keys():
            if demographic_group == "total": continue
            wers[demographic_group] = results[dataset][model][language][demographic_group]
            #if len(wers[demographic_group]) < 10: print(language, demographic_group)

          if gender_disparity:
            gender_sign = do_significance_testing(wers, "female", "male")
            if gender_sign:
              gender_significant_results[model][f"{gender_sign}"] += 1

          if intersectionality_disparity:
            if dataset == "voxpopuli": continue

            # check that both genders have age distributions:
            string_demographic_groups = "+".join(wers.keys())
            if not "female_" in string_demographic_groups and "male_" in string_demographic_groups: continue
            
            intersectionality_groups = [x for x in wers.keys() if "_" in x]
            print(intersectionality_groups)
            for demographic_group in intersectionality_groups:
              intersec_sign = do_significance_testing(wers, demographic_group, "all")
                
              if intersec_sign:
                if intersec_sign == "all": continue
                if demographic_group not in intersectionality_significant_results[model].keys():
                  intersectionality_significant_results[model][f"{demographic_group}"] = 0
                intersectionality_significant_results[model][f"{demographic_group}"] += 1
         
        except KeyError: continue

  if gender_disparity:
    make_plot_genderdiff(gender_significant_results, total_runs)
    visualize_per_model_significance(gender_significant_results, total_runs) 

  if intersectionality_disparity:
    make_heatmap(intersectionality_significant_results, total_runs)

def main(_):
  results = read_results()
  get_significance(results, gender_disparity=True, intersectionality_disparity=True)


if __name__ == '__main__':
  app.run(main)



