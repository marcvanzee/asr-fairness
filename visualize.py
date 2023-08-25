import json
from absl import app
from scipy import stats
from tabulate import tabulate


def read_results():
  with open('results/wers.json', 'r') as f:
    return json.load(f)

def visualize_per_model_significance(significant_results_per_model, nr_runs):
  table = [["Model", "\\female", "\\male", "All", "\\%"]]
  total = ["Total", 0, 0, 0]
  for model in sorted(significant_results_per_model.keys()):
    model_info = [model, len(significant_results_per_model[model]['female']), len(significant_results_per_model[model]['male']), len(significant_results_per_model[model]['female'])+len(significant_results_per_model[model]['male']), round( 100*( len(significant_results_per_model[model]['female'])+len(significant_results_per_model[model]['male']) )/nr_runs,2)]
    table.append(model_info)

    total[1] += len(significant_results_per_model[model]['female'])
    total[2] += len(significant_results_per_model[model]['male'])
    total[3] += round(len(significant_results_per_model[model]['female'])+len(significant_results_per_model[model]['male'])/nr_runs,2)

    #print(f"{model} has {len(significant_results_per_model[model]['female'])+len(significant_results_per_model[model]['male'])} significant results ({len(significant_results_per_model[model]['female'])} female, {len(significant_results_per_model[model]['male'])} male).")
  table.append(total)

  print(tabulate(table, headers="firstrow", tablefmt="latex"))

def get_significance(results):

  significant_results = {"female": 0, "male": 0}
  significant_results_per_model = {}
  nr_runs = 0

  for dataset in results: # cv vp
    for model in results[dataset]: # tiny, small, mms-fl102
      for language in results[dataset][model]: # nl, fr
        nr_runs += 1
        try:
          wers = [results[dataset][model][language]["female"], results[dataset][model][language]["male"]]
          averages = [round(100*sum(wers[0])/len(wers[0]),1), round(100*sum(wers[1])/len(wers[1]),1)]

          sign = stats.ttest_ind(wers[0], wers[1])
          if sign[1] < 0.05:
            if model not in significant_results_per_model.keys():

              significant_results_per_model[model] = {"female": [], "male": []}
            if sign[0] < 0:
              significant_results["male"] += 1
              significant_results_per_model[model]["male"].append(language)
            elif sign[0] > 0:
              significant_results["female"] += 1
              significant_results_per_model[model]["female"].append(language)

        except KeyError: continue
  print(f"There are {nr_runs} runs (dataset/language/model), and in {significant_results['female']} cases ({round(100*significant_results['female']/nr_runs,1)}%), the models perform significantly poorer on women. In {significant_results['male']} cases ({round(100*significant_results['male']/nr_runs,1)}%), the models perform significantly poorer on men.")
  
  visualize_per_model_significance(significant_results_per_model, nr_runs) 




def main(_):
  results = read_results()

  get_significance(results)


  # for model in results["voxpopuli"]:
  #   print("MODEL:", model)
  #   for language in results["voxpopuli"][model]:
  #     print("<<<<<<<<")
  #     print("LANGUAGE:", language)
  #     for demographic_group in results["voxpopuli"][model][language]:
  #       print("DEMO GROUP:", demographic_group)


  #       print(f"There are {len(results['voxpopuli'][model][language][demographic_group])} sentences by {demographic_group}")
  #       print("Average WER:", sum(results['voxpopuli'][model][language][demographic_group])/len(results['voxpopuli'][model][language][demographic_group]))
  #       print()


if __name__ == '__main__':
  app.run(main)

"""

# per model, per dataset
                  cv         |       vp
          m   f   t   stat   |   m   f   t stat    
tiny
large
large-v2
huge
mml


# per langage, per dataset and per model (MANY)
        male    female    total     stat sig
nl
fr
en


# per model, per language, independent of dataset
            tiny                large-v2

nl    m   f   t   stat   |   m   f   t  stat    
fr
en

"""



