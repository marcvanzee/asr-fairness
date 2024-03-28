"""
We compute the wer over a dataset by summing the errors of all points and then
normalize by the sum of sequence lengths. Another option is to take the average
WER over all sentences, but this is heavily skewed by error for short
ground-truth sentences, so we avoid that.
"""
import csv
import collections
import glob
import json
import os

from absl import app

from scipy.stats import shapiro 

import matplotlib.pyplot as plt


EvalInfo = collections.namedtuple('EvalInfo', ['ref_len', 'wer'])

accent_rewrite = {
"España: Norte peninsular (Asturias, Castilla y León, Cantabria, País Vasco, Navarra, Aragón, La Rioja, Guadalajara, Cuenca)": "Iberian",
"España: Norte peninsular (Asturias, Castilla y León, Cantabria, País Vasco, Navarra, Aragón, La Rioja, Guadalajara, Cuenca),catalan": "Iberian",
"España: Sur peninsular (Andalucia, Extremadura, Murcia)": "Iberian",
"España: Centro-Sur peninsular (Madrid, Toledo, Castilla-La Mancha)": "Iberian",
"España: Noroeste Peninsular,Barcelona": "Iberian",
"España: Islas Canarias": "Iberian",
"Estados Unidos": "US",
"Chileno: Chile, Cuyo": "Chilean",
"Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay": "RiverPlatean",
"Andino-Pacífico: Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina": "Andean",
"México": "Mexican",
"Caribe: Cuba, Venezuela, Puerto Rico, República Dominicana, Panamá, Colombia caribeña, México caribeño, Costa del golfo de México": "Caribbean",
"América central": "CentralAmerican",
}

def get_accent(accent_string):
  # if "España:" in accent_string:
  #   return "Iberian"
  # else:
  #   return "Latin"
  if accent_string in accent_rewrite.keys():
    return accent_rewrite[accent_string]
  else:
     print("No matching accent found")

def get_data(filename):
  counter = 0
  with open(filename) as f:
    data = csv.DictReader(f, delimiter='\t')
    for row in data:

      if not row['gender']: continue  # only for commonvoice
      demographic_groups = [row['gender']]
      if not row['accent']: continue  # only for commonvoice
      accent = get_accent(row['accent'])
      demographic_groups.append(accent)
      if 'commonvoice' in filename and row['age'] and row['accent']:
        counter += 1
        demographic_groups.append(f"{row['gender']}_{row['age']}_{accent}")
        demographic_groups.append(f"{row['gender']}_{accent}")
      ref_len = len(row['reference'].split())
      eval_info = EvalInfo(ref_len, float(row['wer']))
      demographic_groups.append("All")
      yield demographic_groups, eval_info
  print(counter)

def save_results(results, directory='results'):
  if not os.path.isdir(directory):
    os.makedirs(directory)
  path = os.path.join(directory, 'eval_data_es.json')
  with open(path, 'w') as f:
     f.write(json.dumps(results, indent=None))
  print('Wrote results to', path)


def main(_):
  languages = []
  results = {}

  for filename in glob.glob("inference_results/*commonvoice_es_*.tsv"):
    elements = filename.split("/")[1].split(".")[0].split("_")
    print('PROCESSING', elements)
    languages.append(elements[3])
    model, modelsize = elements[0], elements[1]
    model_info = f"{model}_{modelsize}"

    results[model_info] = collections.defaultdict(list)
    dem_group_to_eval_data = results[model_info]

    for demographic_groups, eval_info in get_data(filename):
      for demographic_group in demographic_groups:
        dem_group_to_eval_data[demographic_group].append(eval_info)
  save_results(results)


if __name__ == '__main__':
  app.run(main)
