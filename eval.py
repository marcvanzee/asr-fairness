# NB: calculate the WER independently of group. The WER of a ref/inf pair is independent of the other ref/inf pair sentences in a group. 
"""
print("per sentence")
print("nr. 1", wer("women group together", "a woman groups together"))
print("nr. 2", wer("what a weather took a turn", "look the weather took a turn"))
print("average", (wer("women group together", "a woman groups together")+wer("what a weather took a turn", "look the weather took a turn"))/2)

print("across both")
print("average",wer(["women group together", "what a weather took a turn"], ["a woman groups together", "look the weather took a turn"]))

"""
import glob
import csv

from absl import app
from absl import flags
from jiwer import wer, cer, wil


# script 1
# read files
# compare ref and inf
# assign the result to the smallest infra-group (e.g. woman_20)
# write out to pandas database

# script 2
# read db in
# run fairness evaluations

FLAGS = flags.FLAGS


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


def eval_inferred(reference, inferred):
  pass

def process_commonvoice(language, filename):
  file_results = {}

  with open(filename) as f:
    data = csv.reader(f, delimiter='\t')
    print(type(csv))
    for line in data:
      gender, accent, age, locale, reference, reference_original, inferred, inferred_original = line
      print(gender, reference)

def process_voxpopuli(language, filename):
  file_results = {}
  
  with open(filename, 'w+') as f:
    data = csv.reader(f, delimiter='\t')
    for line in data:
      gender, accent, age, locale, reference, reference_original, inferred, inferred_original = line
      print(gender, reference)



  #   if accent is not "None":
  #     print(accent)
  #   minimum_group = f"{gender}"
  #   if infra_group == "gender": continue
  #   if infra_group not in file_results:
  #     file_results[infra_group] = []
  #   file_results[infra_group].append(wer(reference, inferred))

  #   if accent != "None":
  #     print(f"There is an accent: {line}")

  # print('language')
  # total_len = 0
  # total_sum = 0
  # for group in file_results:
  #   s = sum(file_results[group])
  #   l = len(file_results[group])
  #   total_sum += s
  #   total_len += l
  #   print(f"{group} average WER:", s/l)

  # print("Total average WER:", total_sum/total_len)
  #complete_results[dataset][f"{model}_{modelsize}"][language] = [total]


def main(_):

  complete_results = {"commonvoice": {}, "voxpopuli": {}} # dataset/model_modelsize/language = [totalWER, femWER, mascWER]

  filenames = glob.glob("inference_results/*tiny*")

  for filename in filenames:
    elements = filename.split("/")[1].split(".")[0].split("_")
    try:
      model, modelsize, dataset, language, datasetsplit = elements
    except ValueError:
      continue
    if f"{model}_{modelsize}" not in complete_results[dataset]:
      complete_results[dataset][f"{model}_{modelsize}"] = {}

    if dataset == "commonvoice":
      process_commonvoice(language, filename)

    else:
      continue
      #process_voxpopuli(language, filename)
    
if __name__ == '__main__':
  app.run(main)
