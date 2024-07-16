from tqdm import tqdm
import transformers
import torch
import openai
import os
import time
import numpy as np
import time
import wikipedia as wp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def llm_init(model_name):
    global device
    global model
    global tokenizer
    global pipeline

    if model_name == "aya_13b":
        device = "cuda"
        tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
        model = AutoModelForSeq2SeqLM.from_pretrained("CohereForAI/aya-101", device_map="auto")
    
    if model_name == "chatgpt" or model_name == "gpt4":
        openai.api_key = os.getenv("OPENAI_API_KEY")

def wipe_model():
    global device
    global model
    global tokenizer
    global pipeline
    device = None
    model = None
    tokenizer = None
    pipeline = None
    del device
    del model
    del tokenizer
    del pipeline

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200):
    if not model_name == "chatgpt" and not model_name == "gpt4":
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, temperature=temperature, do_sample=True)

        # print(outputs)

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        input_length = inputs.input_ids.shape[1]
        if model_name == "aya_13b" or model_name == "aya23_8b":
            generated_tokens = outputs.sequences[:, 1:-1]
        else:
            generated_tokens = outputs.sequences[:, input_length:]
        generated_text = tokenizer.batch_decode(generated_tokens)[0].strip()
        token_probs = {}
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            token_probs[tokenizer.decode(tok).strip()] = np.exp(score.item())
        if probs:
            return generated_text, token_probs
        else:
            return generated_text
    
    elif model_name == "chatgpt":
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            logprobs=1,
        )
        time.sleep(0.1)
        token_probs = {}
        for tok, score in zip(response.choices[0].logprobs.tokens, response.choices[0].logprobs.token_logprobs):
            token_probs[tok] = np.exp(score)
        if probs:
            return response.choices[0].text, token_probs
        else:
            return response.choices[0].text
    
    elif model_name == "gpt4":
        response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    logprobs=True,
                )
        time.sleep(0.1)
        token_probs = {}
        for thing in response['choices'][0]['logprobs']["content"]:
            token_probs[thing["token"]] = np.exp(thing["logprob"])
        if probs:
            return response['choices'][0]['message']['content'].strip(), token_probs
        else:
            return response['choices'][0]['message']['content'].strip()

def answer_parsing(response):
    # mode 1: answer directly after
    temp = response.strip().split(" ")
    for option in ["A", "B", "C", "D", "E"]:
        if option in temp[0]:
            return option
    # mode 2: "The answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the answer is " + option in temp:
            return option.upper()
    # mode 3: "Answer: A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "answer: " + option in temp:
            return option.upper()
    # mode 4: " A/B/C/D/E " or " A/B/C/D/E."
    for option in ["A", "B", "C", "D", "E"]:
        if " " + option + " " in response or " " + option + "." in response:
            return option
    # mode 5: "The correct answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the correct answer is " + option in temp:
            return option.upper()
    # mode 6: "A: " or "B: " or "C: " or "D: " or "E: "
    for option in ["A", "B", "C", "D", "E"]:
        if option + ": " in response:
            return option
    # mode 7: "A/B/C/D/E" and EOS
    try:
        for option in ["A", "B", "C", "D", "E"]:
            if option + "\n" in response or response[-1] == option:
                return option
    except:
        pass
    # mode 8: "true" and "false" instead of "A" and "B" for feedback abstention

    if "true" in response.lower():
        return "A"
    if "false" in response.lower():
        return "B"

    # fail to parse
    # print("fail to parse answer", response, "------------------")
    return "Z" # so that its absolutely wrong

prompt = "Question: Who is the 44th president of the United States?\nAnswer:"

# llm_init("aya_13b")
# answer = llm_response(prompt, "aya_13b", probs=False)
# print(answer)

text_classifier = None

def mlm_text_classifier(texts, labels, EPOCHS=10, BATCH_SIZE=32, LR=5e-5):
    # train a roberta-base model to classify texts
    # texts: a list of strings
    # labels: a list of labels of 0 or 1

    # load model
    global text_classifier
    text_classifier = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # tokenize
    encodeds = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodeds["input_ids"]
    attention_mask = encodeds["attention_mask"]

    # train
    optimizer = torch.optim.Adam(text_classifier.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    batch_size = BATCH_SIZE
    for epoch in tqdm(range(EPOCHS)):
        for i in range(0, len(input_ids), batch_size):
            optimizer.zero_grad()
            outputs = text_classifier(input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size])
            logits = outputs.logits
            loss = loss_fn(logits, torch.tensor(labels[i:i+batch_size]))
            loss.backward()
            optimizer.step()

def text_classifier_inference(text):
    # provide predicted labels and probability
    # text: a string
    # return: label, probability
    global text_classifier

    assert text_classifier is not None, "text_classifier is not initialized"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text_classifier.eval()
    encodeds = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodeds["input_ids"]
    attention_mask = encodeds["attention_mask"]
    outputs = text_classifier(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return predictions[0].item(), probs[0][predictions[0]].item() # label, probability for the predicted label

# texts = ["I like this movie", "I hate this movie", "I like this movie", "I hate this movie"] * 100
# labels = [1, 0, 1, 0] * 100
# mlm_text_classifier(texts, labels)
# print(text_classifier_inference("I like this movie"))
# print(text_classifier_inference("I hate this movie"))

def get_wiki_summary(text):
    passage = ""
    try:
        for ent in wp.search(text[:100], results = 3):
            try:
                passage = "".join(wp.summary(ent, sentences=10)).replace("\n", " ")
            except:
                # print("error in retrieving summary for " + ent)
                pass
    except:
        print("error in wiki search")
        time.sleep(2)
        pass
    return passage