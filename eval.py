# Skipping locale for now (line[-4])

"""
 NB: calculate the WER independently of group. The WER of a ref/inf pair is independent of the other ref/inf pair sentences in a group. 
print("per sentence")
print("nr. 1", wer("women group together", "a woman groups together"))
print("nr. 2", wer("what a weather took a turn", "look the weather took a turn"))
print("average", (wer("women group together", "a woman groups together")+wer("what a weather took a turn", "look the weather took a turn"))/2)

print("across both")
print("average",wer(["women group together", "what a weather took a turn"], ["a woman groups together", "look the weather took a turn"]))

"""
import csv
import glob
import json

from absl import app
from jiwer import wer


# script 1
# read files
# compare ref and inf
# assign the result to the smallest infra-group (e.g. woman_20)
# write out to pandas database

# script 2
# read db in
# run fairness evaluations


def eval_inferred(reference, inferred):
  pass

def get_data(dataset, filename):

  # Get a list of demographics [female, female20, female20RU] per example
  # Get reference and inferred sentences
  
  with open(filename) as f:
    data = csv.reader(f, delimiter='\t')
    for line in data:
      speaker_demographic_groups = []
      if "commonvoice" in filename:
        demographics = [x for x in [line[0], line[2]] if x != ""]
      else:
        demographics = line[0]
      if len(demographics) < 1: continue
      if "gender" in demographics: continue
      for x in range(len(demographics)):
        speaker_demographic_groups.append('_'.join(demographics[:x+1]))
      reference = line[-4]
      inferred = line[-2]
      yield speaker_demographic_groups, reference, inferred

def save_results(results):
  json_object = json.dumps(results, indent=4)
  with open("results/wers.json", "w") as outfile:
     outfile.write(json_object)

def main(_):

  results = {"commonvoice": {}, "voxpopuli": {}} # dataset/model_modelsize/language = {totalWER: x, femaleWER: y, female20: z}

  filenames = glob.glob("inference_results/*")

  for filename in filenames:
    elements = filename.split("/")[1].split(".")[0].split("_")
    print(f"PROCESSING: {elements}")
    try:
      model, modelsize, dataset, language, datasetsplit = elements
    except ValueError:
      continue

    if f"{model}_{modelsize}" not in results[dataset]:
      results[dataset][f"{model}_{modelsize}"] = {}
    if language not in results[dataset][f"{model}_{modelsize}"]:
      results[dataset][f"{model}_{modelsize}"][language] = {"total": []}

    for speaker_demographic_groups, reference, inferred in get_data(dataset, filename):
      try:
        WER = wer(reference, inferred)
      except ValueError: # some ref or inf are empty strings
        continue
      
      results[dataset][f"{model}_{modelsize}"][language]["total"].append(WER)
      
      for demographic_group in speaker_demographic_groups:
        if demographic_group not in results[dataset][f"{model}_{modelsize}"][language]:
          results[dataset][f"{model}_{modelsize}"][language][demographic_group] = []
        results[dataset][f"{model}_{modelsize}"][language][demographic_group].append(WER)
  save_results(results)
  

    
if __name__ == '__main__':
  app.run(main)
