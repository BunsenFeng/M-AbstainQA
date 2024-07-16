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

        for d in tqdm(data["test"]):
            # original answer correct flag
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False, max_new_tokens=5)
            original_answer = lm_utils.answer_parsing(response)
            # print(response)
            # print(lm_utils.answer_parsing(response))
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)

            # generate a conflicting knowledge passage
            prompt_generate_conflict = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                prompt_generate_conflict += (key + ": " + d["choices"][key] + "\n")
            remaining_choices = list(d["choices"].keys())
            try:
                remaining_choices.remove(original_answer)
            except:
                pass
            wrong_answer = random.choice(remaining_choices)
            prompt_generate_conflict += "Generate a knowledge paragraph about " + wrong_answer + ". The knowledge graph should be in the language of the question."
            response = lm_utils.llm_response(prompt_generate_conflict, model_name, probs=False, temperature=1)

            # answer when presented with conflicting info
            conflict_prompt = "Answer the question with the following knowledge: feel free to ignore irrelevant or wrong information.\n\nKnowledge: " + response + "\n"
            conflict_prompt += "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                conflict_prompt += (key + ": " + d["choices"][key] + "\n")
            conflict_prompt += "Choose one answer from the above choices. The answer is"
            response, probs = lm_utils.llm_response(conflict_prompt, model_name, probs=True, temperature=1, max_new_tokens=5)

            if lm_utils.answer_parsing(response) == original_answer:
                abstain_flags.append(0)
                if original_answer in probs.keys():
                    abstain_scores.append(1-probs[original_answer])
                elif " " + original_answer in probs.keys():
                    abstain_scores.append(1-probs[" " + original_answer])
            else:
                abstain_flags.append(1)
                if lm_utils.answer_parsing(response) in probs.keys():
                    abstain_scores.append(probs[lm_utils.answer_parsing(response)])
                elif " " + lm_utils.answer_parsing(response) in probs.keys():
                    abstain_scores.append(probs[" " + lm_utils.answer_parsing(response)])

    if local_flag:
        with open("preds/" + model_name + "_" + dataset + "_" + speak + "_conflict.json", "w") as f:
            json.dump({"correct_flags": correct_flags, "abstain_flags": abstain_flags, "abstain_scores": abstain_scores}, f)
    
    assert len(correct_flags) == len(abstain_flags)
    print("------------------")
    print("Approach: conflict")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Language:", speak)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")