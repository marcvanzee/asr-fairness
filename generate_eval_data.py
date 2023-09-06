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


EvalInfo = collections.namedtuple('EvalInfo', ['ref_len', 'wer', 'date'])


def get_data(filename):
  with open(filename) as f:
    data = csv.DictReader(f, delimiter='\t')
    for row in data:
      if not row['gender']: continue  # only for commonvoice
      demographic_groups = [row['gender']]
      if 'commonvoice' in filename and row['age']:
        demographic_groups.append(row['gender'] + '_' + row['age'])
      eval_info = EvalInfo(len(row['reference']), float(row['wer']), row['date'])
      yield demographic_groups, eval_info

def save_results(results, directory='results'):
  if not os.path.isdir(directory):
    os.makedirs(directory)
  path = os.path.join(directory, 'eval_data.json')
  with open(path, 'w') as f:
     f.write(json.dumps(results, indent=None))
  print('Wrote results to', path)


def main(_):
  # dataset/model_modelsize/language = {totalWER: x, femaleWER: y, female20: z}
  results = {
    'commonvoice': collections.defaultdict(dict),
    'voxpopuli': collections.defaultdict(dict)
  }

  for filename in glob.glob("inference_results/*.tsv"):
    elements = filename.split("/")[1].split(".")[0].split("_")
    print('PROCESSING', elements)
    model, modelsize, dataset, language, _ = elements
    model_info = f"{model}_{modelsize}"

    results[dataset][model_info][language] = collections.defaultdict(list)
    dem_group_to_eval_data = results[dataset][model_info][language]

    for demographic_groups, eval_info in get_data(filename):
      # store all eval data in the 'all' field.
      dem_group_to_eval_data['all'].append(eval_info)
      for demographic_group in demographic_groups:
        dem_group_to_eval_data[demographic_group].append(eval_info.wer)
  save_results(results)


if __name__ == '__main__':
  app.run(main)
