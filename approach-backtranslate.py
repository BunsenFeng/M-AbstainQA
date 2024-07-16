import json
import argparse
import lm_utils
import metrics
import random
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "aya_13b", "chatgpt", "gpt4"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "mmlu", "hellaswag", "belebele"
    argParser.add_argument("-s", "--speak", help = "speak which language") # "es", "nl", etc.
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-l", "--local", default = False, help="local copy of preds saved")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    speak = args.speak
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

        for d in tqdm(data["test"]):
            # correct flags
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

            # translation

            translate_prompt = original_prompt + "\n\nTranslate the above question to English."
            translated_question = lm_utils.llm_response(translate_prompt, model_name, probs=False, max_new_tokens=512)
            
            translated_answer = lm_utils.llm_response(translated_question + "\n\nAnswer:", model_name, probs=False, max_new_tokens=5)
            reflect_prompt = translated_question + "\n\nAnswer: " + translated_answer + "\n\nThe above answer is:\nA. True\nB. False\nThe answer is" # make abstain decision in English with self-reflect
            response = lm_utils.llm_response(reflect_prompt, model_name, probs=False, max_new_tokens=5)
            if lm_utils.answer_parsing(response) == "A":
                abstain_flags.append(0)
            else:
                abstain_flags.append(1)

    abstain_scores = None

    if local_flag:
        with open("preds/" + model_name + "_" + dataset + "_" + speak + "_backtranslate.json", "w") as f:
            json.dump({"correct_flags": correct_flags, "abstain_flags": abstain_flags, "abstain_scores": abstain_scores}, f)

    print("------------------")
    print("Approach: backtranslate")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Language:", speak)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")