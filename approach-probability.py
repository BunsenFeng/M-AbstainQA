import json
import argparse
import lm_utils
import metrics
import random
import torch.nn
from tqdm import tqdm

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

        correct_labels_dev = []
        option_to_ind = {"A": 0, "B": 1, "C": 2, "D": 3}
        probs = []
        target = []
        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            target.append(option_to_ind[correct_answer])
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response, token_probs = lm_utils.llm_response(original_prompt, model_name, probs=True, max_new_tokens=5)
            # print(response, token_probs)
            # print("------------------")
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_labels_dev.append(1)
            else:
                correct_labels_dev.append(0)
            prob_max = None
            chosen_option = None
            try:
                for token in token_probs.keys():
                    if token.strip() in option_to_ind.keys():
                        prob_max = token_probs[token]
                        chosen_option = token.strip()
                        break
            except:
                # print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            if chosen_option == None:
                # print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            prob_distribution = [0, 0, 0, 0]
            prob_distribution[option_to_ind[chosen_option]] = prob_max
            # evenly split between other options
            for i in range(4):
                if i != option_to_ind[chosen_option]:
                    prob_distribution[i] = (1 - prob_max) / 3
            probs.append(prob_distribution)

        # determine optimal threshold for abstain

        prob_maximum = max([max(prob) for prob in probs])
        prob_minimum = min([max(prob) for prob in probs])

        min_error = 1e6
        best_threshold = 0
        for threshold in range(1, 100):

            # no 100% abstain or 100% answer
            if threshold / 100.0 >= prob_maximum or threshold / 100.0 <= prob_minimum:
                continue

            error = 0
            abstain_cnt = 0
            for i in range(len(correct_labels_dev)):
                if max(probs[i]) < float(threshold/100.0):
                    abstain_cnt += 1
                    if correct_labels_dev[i] == 1:
                        error += 1
                else:
                    if correct_labels_dev[i] == 0:
                        error += 1
            if error < min_error and abstain_cnt / len(correct_labels_dev) < 0.5: # abstain rate less than 50%
                min_error = error
                best_threshold = float(threshold/100.0)
                # print("best threshold:", best_threshold)
                # print("best error:", min_error)

        # obtain abstain flags for test set

        for d in tqdm(data["test"]):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response, token_probs = lm_utils.llm_response(original_prompt, model_name, probs=True, max_new_tokens=5)
            prob_max = None
            chosen_option = None
            try:
                for token in token_probs.keys():
                    if token.strip() in option_to_ind.keys():
                        prob_max = token_probs[token]
                        chosen_option = token.strip()
                        break
            except:
                # print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            if chosen_option == None:
                # print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            prob_distribution = [0, 0, 0, 0]
            prob_distribution[option_to_ind[chosen_option]] = prob_max
            # evenly split between other options
            for i in range(4):
                if i != option_to_ind[chosen_option]:
                    prob_distribution[i] = (1 - prob_max) / 3

            if prob_distribution[option_to_ind[chosen_option]] < best_threshold:
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)
            abstain_scores.append(1-prob_distribution[option_to_ind[chosen_option]])

    # print(correct_flags)
    # print(abstain_flags)
    # print(abstain_scores)

    if local_flag:
        with open("preds/" + model_name + "_" + dataset + "_" + speak + "_probability.json", "w") as f:
            json.dump({"correct_flags": correct_flags, "abstain_flags": abstain_flags, "abstain_scores": abstain_scores}, f)

    print("------------------")
    print("Approach: probability")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Language:", speak)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")