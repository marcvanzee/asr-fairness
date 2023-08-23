# File format
All results are stored as tab separated values in the following files:

`[model]_[model_size]_[dataset_name]_[split]_[lang].tsv`

## Common Voice
https://huggingface.co/datasets/mozilla-foundation/common_voice_9_0

### Original dataset structure

```
example
{
  'client_id': 'd59478fbc1ee646a28a3c652a119379939123784d99131b865a89f8b21c81f69276c48bd574b81267d9d1a77b83b43e6d475a6cfc79c232ddbca946ae9c7afc5', 
  'path': 'et/clips/common_voice_et_18318995.mp3', 
  'audio': {
    'path': 'et/clips/common_voice_et_18318995.mp3', 
    'array': array([-0.00048828, -0.00018311, -0.00137329, ...,  0.00079346, 0.00091553,  0.00085449], dtype=float32), 
    'sampling_rate': 48000
  }, 
  'sentence': 'Tasub kokku saada inimestega, keda tunned juba ammust ajast saati.', 
  'up_votes': 2, 
  'down_votes': 0, 
  'age': 'twenties', 
  'gender': 'male', 
  'accent': '', 
  'locale': 'et', 
  'segment': ''
}
```

### CommonVoice results columns in TSV files

| column_name | meaning |
|-------------|---------|
|reference   |   		`example['sentence']` after whisper normalization |
reference_original |  	`example['sentence']`
inferred  |            inferred sentence after whisper normalization
inferred_original	| inferred sentence
age           |        `example['age']`
gender        |        `example['gender']` 
accent        |        `example['accent']`
locale         |       `example['locale']`


## VOXPOPULI

https://huggingface.co/datasets/facebook/voxpopuli

### Original dataset structure

```
example
{
  'audio_id': '20180206-0900-PLENARY-15-hr_20180206-16:10:06_5',
  'language': 11,  # "hr"
  
  'raw_text': '',
  'normalized_text': 'poast genitalnog sakaenja ena u europi tek je jedna od manifestacija takve tetne politike.',
  'gender': 'female',
  'speaker_id': '119431',
  'is_gold_transcript': True,
  'accent': 'None'
}
```

### VoxPopuli results columns in TSV files

| column_name | meaning |
|-------------|---------|
| reference   | `example['normalized_text']` after normalization |
| reference_original	| `example['normalized_text']`
| inferred  | inferred  sentence after normalization |
| inferred_original	| inferred sentence |
| gender       |             	`example['gender']`
| speaker_id    |            	`example['speaker_id']`
| accent         |           	`example['accent']`

