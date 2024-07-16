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
    argParser.add_argument("-f", "--feedback", default = False, help ="whether to save generated feedbacks in a seperate file")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    speak = args.speak
    approach_type = "self"
    portion = args.portion
    local_flag = args.local
    feedback_flag = args.feedback

    lm_utils.llm_init(model_name)

    correct_flags = []
    abstain_flags = []
    abstain_scores = []

    with open("data/" + dataset + "/" + dataset + "_" + speak + ".json", "r") as f:

        data = json.load(f)

        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]

        answers = []
        feedback_1 = []
        feedback_2 = []
        feedback_3 = []
        
        # obtain correct flags
            
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
            answers.append(response)

        # obtain feedbacks

        prompt_feedback_list = []

        for d, i in tqdm(zip(data["test"], range(len(data["test"])))):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. Proposed answer:"

            prompt_feedback = original_prompt + " " + answers[i].strip() + "\nPlease review the proposed answer and provide a paragraph of feedback on its correctness. Feedback should be in the language of the question.\nFeedback:"
            prompt_feedback_list.append(prompt_feedback)
        
        if approach_type == "self": # expert reviewers
            for prompt_feedback in tqdm(prompt_feedback_list):
                # generate knowledge from different expert's perspectives: facts, multi-hop, and commonsense
                prompt_feedback_experts = []
                for domain_name in ["factual information", "multi-hop reasoning", "commonsense knowledge"]:
                    expert_prompt = "Generate some knowledge about the question, focusing on " + domain_name + ". Knowledge should be in the language of the question.\n" + prompt_feedback.split("\n")[0] + "\nKnowledge:"
                    prompt_feedback_experts.append("Knowledge: " + lm_utils.llm_response(expert_prompt, model_name, probs=False, temperature=1).split("\n")[0].strip() + "\n" + prompt_feedback)
                assert len(prompt_feedback_experts) == 3

                response = lm_utils.llm_response(prompt_feedback_experts[0], model_name, probs=False, temperature=1)
                response = response.split("\n")[0].strip()
                if len(response) == 0: # to avoid no generated feedback in multilingual settings
                    response = "No feedback provided."
                feedback_1.append(response)
                response = lm_utils.llm_response(prompt_feedback_experts[1], model_name, probs=False, temperature=1)
                response = response.split("\n")[0].strip()
                if len(response) == 0:
                    response = "No feedback provided."
                feedback_2.append(response)
                response = lm_utils.llm_response(prompt_feedback_experts[2], model_name, probs=False, temperature=1)
                response = response.split("\n")[0].strip()
                if len(response) == 0:
                    response = "No feedback provided."
                feedback_3.append(response)
        elif approach_type == "others":
            raise NotImplementedError("Not implemented for others approach")
        
        # obtain abstain flags and scores

        prompt_area_chair_list = []
        assert len(data["test"]) == len(answers) == len(feedback_1) == len(feedback_2) == len(feedback_3)
        for i in range(len(data["test"])):
            d = data["test"][i]
            prompt_area_chair = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                prompt_area_chair += (key + ": " + d["choices"][key] + "\n")

            prompt_area_chair += "Choose one answer from the above choices. Proposed answer: " + answers[i].strip() + "\n\nFeedback 1: " + feedback_1[i].strip() + "\n\nFeedback 2: " + feedback_2[i].strip() + "\n\nFeedback 3: " + feedback_3[i].strip() + "\n\nBased on the feedback, is the proposed answer True or False?"
            prompt_area_chair_list.append(prompt_area_chair)
        assert len(prompt_area_chair_list) == len(data["test"])

        if approach_type == "self":
            for prompt_area_chair in tqdm(prompt_area_chair_list):
                response, probs = lm_utils.llm_response(prompt_area_chair, model_name, probs=True, max_new_tokens=10)

                if "true" in response.lower():
                    abstain_flags.append(0)
                elif "false" in response.lower():
                    abstain_flags.append(1)
                else:
                    # print("Error: abstain flag not found")
                    abstain_flags.append(random.randint(0, 1))
                try:
                    temp = -1
                    for key in probs.keys():
                        if key.lower() == "true" or key.lower() == " true":
                            temp = 1 - probs[key]
                            break
                        elif key.lower() == "false" or key.lower() == " false":
                            temp = probs[key]
                            break
                    if temp == -1:
                        abstain_scores.append(0.5)
                    else:
                        abstain_scores.append(temp)
                except:
                    abstain_scores.append(0.5)
        elif approach_type == "others":
            raise NotImplementedError("Not implemented for others approach")

        if feedback_flag:
            generated_feedback_list = []
            assert len(feedback_1) == len(feedback_2) == len(feedback_3) == len(abstain_flags)
            for i in range(len(feedback_1)):
                d = data["test"][i]
                question_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    question_prompt += (key + ": " + d["choices"][key] + "\n")

                generated_feedback_list.append({
                    "question": question_prompt,
                    "proposed_answer": answers[i],
                    "feedbacks": [feedback_1[i], feedback_2[i], feedback_3[i]],
                    "abstain_flag": abstain_flags[i],
                    "correct_flag": correct_flags[i],
                })

            with open("feedbacks/" + model_name + "_" + dataset + "_" + speak + "_" + approach_type + "_mononative.json", "w") as f:
                json.dump(generated_feedback_list, f, indent=4)


    # print(abstain_scores)

    if local_flag:
        with open("preds/" + model_name + "_" + dataset + "_" + speak + "_mononative.json", "w") as f:
            json.dump({"correct_flags": correct_flags, "abstain_flags": abstain_flags, "abstain_scores": abstain_scores}, f)
        
    print("------------------")
    print("Approach: mononative")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Language:", speak)
    print("Type:", approach_type)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")