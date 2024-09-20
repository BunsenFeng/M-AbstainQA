# Multilingual AbstainQA Repository

This is the official repo for [Teaching LLMs to Abstain across Languages via Multilingual Feedback](https://arxiv.org/abs/2406.15948) @ EMNLP 2024.

### Environment

```
conda env create -f abstainqa.yaml
conda activate abstainqa
export OPENAI_API_KEY="YOUR_KEY"
```

### Methods

We provide the implementation of 13 baselines and proposed approaches in the paper. Each `approach-<name>.py` file contains the implementation of the corresponding approach. Shared parameters for each approach:

```
-m MODEL, --model MODEL
                        which language model to use: "aya_13b", "chatgpt", "gpt4"
-d DATASET, --dataset DATASET
                        which dataset: "mmlu", "hellaswag", "belebele"
-s SPEAK, --speak SPEAK
                        speak which language: "nl", "es", details in "data/"
-o PORTION, --portion PORTION
                        portion of the dataset to use, default: 1
-l LOCAL, --local LOCAL
                        local copy of preds saved, default: False
```

These are the default models and datasets we provide in the implementation: more on adding your own later. Portion (0-1) means only evaluating on the first `x%` of the dataset in case the LLM is large and evaluation is slow. We introduce the methods in the following:

#### Calibration: `approach-probability.py`

The `Probs` approach in Table 1.

```
approach-probability.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL]
```

#### Calibration: `approach-temperature.py`

The `Temp` approach in Table 1.

```
approach-temperature.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL]
```

#### Calibration: `approach-askcalibrate.py`

The `Ask Cali.` approach in Table 1.

```
approach-askcalibrate.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL]
```

#### Training: `approach-instructiontune.py`

The `Instruct` approach in Table 1.

```
approach-instructiontune.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-p PHASE] [-t TUNED_MODEL_NAME] [-o PORTION] [-l LOCAL]

options:
  -s SETTING, --setting SETTING
                        generate or evaluate
  -t TUNED_MODEL_NAME, --tuned_model_name TUNED_MODEL_NAME
                        name of the tuned model, either chatgpt via OpenAI API or local/hf copy of tuned model path
```

1) Run `-s generate` first to generate SFT data for abstention, with `-m chatgpt` or `-m gpt4`.
2) SFT the model (`chatgpt` or `gpt4`): do it on your own with the OpenAI API.
3) Run `-s evaluate` with `-t <tuned_model>`, OpenAI model ID for chatgpt/gpt4.

#### Prompting: `approach-reflect.py`

The `Reflect` approach in Table 1.

```
approach-reflect.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL]
```

#### Prompting: `approach-moreinfo.py`

The `MoreInfo` approach in Table 1.

```
approach-moreinfo.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL]
```

#### Prompting: `approach-backtranslate.py`

The `BackTrans` approach in Table 1.

```
approach-backtranslate.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL]
```

#### Consistency: `approach-scthreshold.py`

The `SCthres.` approach in Table 1.

```
approach-scthreshold.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-p PATH] [-o PORTION] [-l LOCAL]
options:
  -p PATH, --path PATH  number of paths to use for self consistency, default: 5
```

#### Consistency: `approach-conflict.py`

The `Conflict` approach in Table 1.

```
approach-conflict.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL]
```

#### Ours: `approach-mononative.py`

The `monolingual, native` approach in Section 2.

```
approach-mononative.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL] [-f FEEDBACK]

options:
  -f FEEDBACK, --feedback FEEDBACK
                        whether to save generated feedbacks in a seperate file in feedbacks/, default: False
```

The next few approaches share the same `-f` option.

#### Ours: `approach-monoenglish.py`

The `monolingual, English` approach in Section 2.

```
approach-monoenglish.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL] [-f FEEDBACK]
```

#### Ours: `approach-multirandom.py`

The `multilingual, random` approach in Section 2.

```
approach-multirandom.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL] [-f FEEDBACK]
```

#### Ours: `approach-multirelated.py`

The `multilingual, related` approach in Section 2.

```
approach-multirelated.py [-h] [-m MODEL] [-d DATASET] [-s SPEAK] [-o PORTION] [-l LOCAL] [-f FEEDBACK]
```

### Models

`lm_utils.py` provides inference code for `aya_13b`, `chatgpt`, and `gpt4`. If you want to add new models, add it in both `lm_init()` where you initialize the model and tokenizer; and `llm_response()` where you generate text with it and provide token probabilities (if any). Make sure that your model is truly multilingual so that it supports long-tail languages like Nepali and Telugu.

### Datasets

We provide datasets in `data/` for `mmlu`, `hellaswag`, `belebele`. The first two comes from [link](https://github.com/nlp-uoregon/mlmm-evaluation) and the third from [link](https://arxiv.org/abs/2308.16884). We sample a fixed size of validation and test sets across languages. We consider 26 languages, please check out [link](https://github.com/nlp-uoregon/mlmm-evaluation) for details. If you want to add new datasets, add it in `data/` and follow the same format as the existing ones. These datasets are multiple-choice QA datasets, while we plan to support non-MC datasets in future work.

### Metrics

`metrics.py` provides the implementation of AbstainQA metrics (Section 3) calcualted from `correct_flags`, `abstain_flags`, and `abstain_scores` (if any). Feel free to add your AbstainQA metric and add it to the return dictionary.

### Citation

```
@article{feng2024teaching,
  title={Teaching LLMs to Abstain across Languages via Multilingual Feedback},
  author={Feng, Shangbin and Shi, Weijia and Wang, Yike and Ding, Wenxuan and Ahia, Orevaoghene and Li, Shuyue Stella and Balachandran, Vidhisha and Sitaram, Sunayana and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2406.15948},
  year={2024}
}
```
