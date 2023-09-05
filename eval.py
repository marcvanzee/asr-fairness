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
from jiwer import wer


def get_data(filename):
  # Get a list of demographics [female, female20] per example
  
  with open(filename) as f:
    data = csv.DictReader(f, delimiter='\t')
    for row in data:
      if not row['gender']: continue  # only for commonvoice
      demographic_groups = [row['gender']]
      if 'commonvoice' in filename and row['age']:
        demographic_groups.append(row['gender'] + '_' + row['age'])
      yield demographic_groups, row['reference'], row['inferred']

def save_results(results, directory='wer_results'):
  if not os.path.isdir(directory):
    os.makedirs(directory)
  path = os.path.join(directory, 'wers.join')
  with open(path, 'w') as f:
     f.write(json.dumps(results, indent=0))
  print('Wrote results to', path)

def main(_):
  # dataset/model_modelsize/language = {totalWER: x, femaleWER: y, female20: z}
  results = {
    'commonvoice': collections.defaultdict(dict),
    'voxpopuli': collections.defaultdict(dict)
  }

  for filename in glob.glob("inference_results/*.tsv")[:2]:
    elements = filename.split("/")[1].split(".")[0].split("_")
    print('PROCESSING', elements)
    model, modelsize, dataset, language, _ = elements

    key = f"{model}_{modelsize}"
    results[dataset][key][language] = collections.defaultdict(list)

    for demographic_groups, reference, inferred in get_data(filename):
      try:
        wer_score = wer(reference, inferred)
      except ValueError: # some ref or inf are empty strings
        continue
      
      for demographic_group in demographic_groups:
        results[dataset][key][language][demographic_group].append(wer_score)
  save_results(results)


if __name__ == '__main__':
  app.run(main)
