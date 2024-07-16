import os
import json
import argparse
import lm_utils
import metrics
import random
import time
import torch
import openai
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

if __name__ == "__main__":

    openai.api_key = os.getenv("OPENAI_API_KEY")

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "aya_13b", "chatgpt", "gpt4"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "mmlu", "hellaswag", "belebele"
    argParser.add_argument("-s", "--speak", help="speak which language") # "nl", "es", etc.
    argParser.add_argument("-p", "--phase", help="generate or evaluate") # "generate" for generating instruction-tuning dataset, "evaluate" if evaluating this approach with a tuned model
    argParser.add_argument("-t", "--tuned_model_name", default = None, help="name of the tuned model") # tuned model name
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-l", "--local", default = False, help="local copy of preds saved")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    speak = args.speak
    setting = args.phase
    tuned_model_name = args.tuned_model_name
    portion = args.portion
    local_flag = args.local

    lm_utils.llm_init(model_name)

    correct_flags = []
    abstain_flags = []
    abstain_scores = []

    with open("data/" + dataset + "/" + dataset + "_" + speak + ".json", "r") as f:
        data = json.load(f)

        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]

        # obtain correct flags for test set

        if setting == "evaluate":
            for d in tqdm(data["test"]):
                original_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    original_prompt += (key + ": " + d["choices"][key] + "\n")
                original_prompt += "Choose one answer from the above choices. The answer is"
                response = lm_utils.llm_response(original_prompt, model_name, probs=False, max_new_tokens=5)
                # print(response)
                # print(lm_utils.answer_parsing(response))
                if lm_utils.answer_parsing(response) == d["answer"]:
                    correct_flags.append(1)
                else:
                    correct_flags.append(0)

        # create instruction tuning dataset based on the dev set

        if setting == "generate":
            texts = []
            for d in tqdm(data["dev"]):
                original_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    original_prompt += (key + ": " + d["choices"][key] + "\n")
                original_prompt += "Choose one answer from the above choices. The answer is"
                response = lm_utils.llm_response(original_prompt, model_name, probs=False, max_new_tokens=5)
                correct_flag = None
                if lm_utils.answer_parsing(response) == d["answer"]:
                    correct_flag = 1
                else:
                    correct_flag = 0

                # generate explanations
                new_prompt = original_prompt + response + "\nExplanation:"
                explanation = lm_utils.llm_response(new_prompt, model_name, probs=False)
                
                if correct_flag:
                    texts.append({"messages": [{"role": "system", "content": "Answer the following question. If you don't have enough knowledge, abstain by saying 'sorry, I don't have enough knowledge to answer this question.'"}, {"role": "user", "content": original_prompt}, {"role": "assistant", "content": response + "\n" + explanation}]})
                else:
                    texts.append({"messages": [{"role": "system", "content": "Answer the following question. If you don't have enough knowledge, abstain by saying 'sorry, I don't have enough knowledge to answer this question.'"}, {"role": "user", "content": original_prompt}, {"role": "assistant", "content": "sorry, I don't have enough knowledge to answer this question."}]})
                
            if not model_name == "chatgpt" or not model_name == "gpt4":
                texts_new = []
                for text in texts:
                    texts_new.append({"instruction": text["messages"][0]["content"], "input": text["messages"][1]["content"], "output": text["messages"][2]["content"]})
                texts = texts_new

            # write texts in a jsonline format
            with open("data/" + dataset + "-" + model_name + "-" + speak + "-instruction-tuning.jsonl", "w") as f:
                for text in texts:
                    f.write(json.dumps(text) + "\n")

        # getting abstain flags with the instruction-tuned version of ChatGPT

        if setting == "evaluate":
            if model_name == "aya_13b":
                raise NotImplementedError("Instruction tuning is not supported for Aya-13B")
            if model_name == "chatgpt" or model_name == "gpt4":
                assert tuned_model_name is not None

                # # change to OpenAI API
                # openai.api_type = 'open_ai'
                # openai.api_base = 'https://api.openai.com/v1'
                # openai.api_version = None

                for d in tqdm(data["test"]):
                    original_prompt = "Question: " + d["question"] + "\n"
                    for key in d["choices"].keys():
                        original_prompt += (key + ": " + d["choices"][key] + "\n")
                    original_prompt += "Choose one answer from the above choices. The answer is"

                    try:
                        completion = openai.ChatCompletion.create(
                            model=tuned_model_name,
                            messages=[
                                {"role": "system", "content": "Answer the following question. If you don't have enough knowledge, abstain by saying 'sorry, I don't have enough knowledge to answer this question.'"},
                                {"role": "user", "content": original_prompt}
                            ],
                            max_tokens=200,
                        )
                        time.sleep(0.1)
                        response = completion.choices[0].message["content"]
                    except:
                        # failed to generate valid text, just abstain!
                        response = "sorry"
                    if "sorry" in response.lower():
                        abstain_flags.append(1)
                    else:
                        abstain_flags.append(0)
        
        # fine-tuned API models do not support probs, atm
        abstain_scores = None

    if local_flag:
        with open("preds/" + model_name + "_" + dataset + "_" + speak + "_instructiontune.json", "w") as f:
            json.dump({"correct_flags": correct_flags, "abstain_flags": abstain_flags, "abstain_scores": abstain_scores}, f)

    if setting == "evaluate":
        print("------------------")
        print("Approach: instructiontune")
        print("Model:", model_name)
        print("Dataset:", dataset)
        print("Language:", speak)
        print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
        print("------------------")