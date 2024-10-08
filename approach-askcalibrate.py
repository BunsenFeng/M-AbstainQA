import json
import argparse
import lm_utils
import metrics
import random
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "aya_13b", "chatgpt", "gpt4"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "mmlu", "hellaswag", "belebele"
    argParser.add_argument("-s", "--speak", help="speak which language") # "nl", "es", etc.
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

        # portion
        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]
        
        # obtain correct flags for test set

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

        # obtain threshold of asked-for probability for abstain

        correct_labels_dev = []
        probabilties_dev = []

        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name, max_new_tokens=5)
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_labels_dev.append(1)
            else:
                correct_labels_dev.append(0)

        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            original_prompt = "Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nThe question is: "
            original_prompt += (d["question"] + "\n")
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. Guess:"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False, max_new_tokens=5)
            # print("------------------")
            # print(response)
            original_prompt += " " + response + "\n"
            original_prompt += "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\nProbability:"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False, temperature=1, max_new_tokens=10)
            # print(response)
            # print("------------------")
            prob = None
            num_list = re.findall("\d+\.\d+", response) + ["0" + a for a in re.findall(".\d+", response)]
            for num in num_list:
                try:
                    temp = float(num)
                except:
                    continue
                if temp >= 0 and temp <= 1:
                    prob = temp
                    break
            if prob == None:
                # print("prob is not found!")
                prob = 0.5
            assert prob >= 0 and prob <= 1
            probabilties_dev.append(prob)

        # determine optimal threshold based on F1-score
        best_threshold = 0
        best_f1_score = 0
        for threshold in range(1, 100):
            abstain_flags = []
            for i in range(len(correct_labels_dev)):
                if probabilties_dev[i] < float(threshold/100.0):
                    abstain_flags.append(1)
                else:
                    abstain_flags.append(0)
            f1 = f1_score(correct_labels_dev, abstain_flags)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = float(threshold/100.0)
                print("best threshold:", best_threshold)
        
        # obtain abstain flags for test set

        abstain_flags = []

        for d in tqdm(data["test"]):
            correct_answer = d["answer"]
            original_prompt = "Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nThe question is: "
            original_prompt += (d["question"] + "\n")
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. Guess:"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False, max_new_tokens=5)
            # print("------------------")
            # print(response)
            original_prompt += " " + response + "\n"
            original_prompt += "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\nProbability:"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False, temperature=1, max_new_tokens=10)
            # print(response)
            # print("------------------")
            prob = None
            num_list = re.findall("\d+\.\d+", response)
            for num in num_list:
                try:
                    temp = float(num)
                except:
                    continue
                if temp >= 0 and temp <= 1:
                    prob = temp
                    break
            if prob == None:
                # print("prob is not found!")
                prob = 0.5
            assert prob >= 0 and prob <= 1
            if prob < best_threshold:
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)
            abstain_scores.append(1-prob)

    # print(correct_flags)
    # print(abstain_flags)
    # print(abstain_scores)

    if local_flag:
        with open("preds/" + model_name + "_" + dataset + "_" + speak + "_askcalibrate.json", "w") as f:
            json.dump({"correct_flags": correct_flags, "abstain_flags": abstain_flags, "abstain_scores": abstain_scores}, f)

    print("------------------")
    print("Approach: askcalibrate")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Language:", speak)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")