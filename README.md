# LLM Eval Suite for African Languages

## Setup

This repo is Poetry compatible. To install the dependencies, ensure you have Poetry installed and run:

```bash
poetry install
```

## Description

This project is aimed at creating a standard evaluation library and scripts for assessing the performance of language models on tasks related to African languages. This project uses the [lm-evaluation-harness by EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation, focusing on African language tasks. Some of the tasks contained in this leaderboard have already been built into the harness (e.g. IrokoBench Tasks & Belebele) but we make some slight modifications to some tasks such as 

- Improving the formatting of the prompts because language models have been shown to be sensitive to the formatting of the prompts ([Sclar et al.](https://arxiv.org/pdf/2310.11324)). Specifically, we strip the prompt of excess whitespace and whitelines and next-line characters.

- Making the leaderboard reproducible by fixing the seeds for generating the fewshot examples needed for the tasks.

- Explore a scenario where the system prompts / prompt templates are written in the language being evaluated other than English.

## Tasks

### Multiple Choice Tasks (Accuracy)

Multiple choice tasks are tasks where the model is given a prompt and a list of options to choose from. The model is expected to select the correct option from the list of options. The tasks that we evaluate in this Category are:

- AfriMMLU : AfriMMLU ([Adelani et al.](https://arxiv.org/pdf/2406.03368)) is a multilingual multi-choice language understanding benchmark in 16 languages of focus across a variety of topics. The task is to predict the correct answer to a multiple-choice question given a context and a list of possible answers.

    Sample from the dataset in Yoruba using the prompts provided by the authors:
    ```
    You are a highly knowledgeable and intelligent artificial intelligence model answers multiple-choice questions about elementary_mathematics

    Question: Kínni iyì p nínú 24 = 2p?

    Choices:
        A: p = 4
        B: p = 8
        C: p = 12
        D: p = 24
    ```


- AfriXNLI : 
- BeleBele :

### Mathematics Reasoning (Exact Match)

...

### Text Classification (F1 Score)

...

### Token Classification (F1 Score)

...

### Question Answering (Rouge)

...

### Summarization (Rouge)

...

## How does LM-Harness evaluate multiple choice tasks?

For multiple-choice questions, in the log-probability mode, the framework creates a set of input and output pairs using the input prompt and check the probablity of generation of each of the choices. The choice with the highest probability is selected as the answer. The framework selects the choice with the highest probability as the answer and checks if it is the correct answer. This approach is [detailed in this blog post](https://blog.eleuther.ai/multiple-choice-normalization/)

For example, using the sample AfriMMLU prompt above, the framework will generate the following input-output pairs:

```
input_prompt =  ...
choices = ["p = 4", "p = 8", "p = 12", "p = 24"]

inputs = [('input_prompt', 'p = 4'), ('input_prompt', 'p = 8'), ('input_prompt', 'p = 12'), ('input_prompt', 'p = 24')]
```

For each of these inputs, the framework will generate the log-probability of the choice being the correct answer. The choice with the highest probability is selected as the answer. This is a sample output of logits from lm-eval

```
[[(-16.883224487304688, False)], [(-14.883224487304688, False)], [(-12.008225440979004, False)], [(-16.508224487304688, False)]]
```

A minimalist example of how you would do this in pytorch is shown below:

<details>
<summary>Click to expand</summary>

```python
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "'You are a highly knowledgeable and intelligent artificial intelligence\nmodel answers multiple-choice questions about elementary_mathematics\n\nQuestion: Kínni iyì p nínú 24 = 2p?\n\nChoices:\n        A: p = 4\n        B: p = 8\n        C: p = 12\n        D: p = 24\n\nAnswer:  "
choices = ["p = 4", "p = 8", "p = 12", "p = 24"]

## Encode the input prompt and choices
input_tokens = tokenizer.encode(prompt, return_tensors="pt")
choice_tokens = [tokenizer.encode(choice, return_tensors="pt") for choice in choices]

## Generate the logits for each choice
choice_probablities = {}

for i, choice in enumerate(choice_tokens):

    current_input_tokens = input_tokens.clone()
    curr_choice_prob = 1.0
    for token in choice[0]:
        
        # run the prompt through the model
        with torch.no_grad():
            outputs = model(current_input_tokens)
        
        # get the logits for the last token and the probability of the token
        last_token_logits = outputs.logits[0, -1, :]
        token_prob = torch.nn.functional.softmax(last_token_logits, dim=0)[token].item()
        curr_choice_prob*= token_prob

        # update the input tokens
        current_input_tokens = torch.cat([current_input_tokens, torch.tensor([token]).unsqueeze(0)], dim=1)
    
    # average the logits for the choice
    choice_probablities[choices[i]] = curr_choice_prob


# Print probabilities and select the highest
print("Probabilities for each choice:")
for choice, prob in choice_probablities.items():
    print(f"{choice}: {prob:.20f}")

# Select the choice with the highest probability
best_choice = max(choice_probablities, key=choice_probablities.get)
print(f"\nThe choice with the highest probability is: {best_choice}")
```
```
Probabilities for each choice:
p = 4: 0.00000000000000176193
p = 8: 0.00000000000001162337
p = 12: 0.00000000000031213047
p = 24: 0.00000000000000706066

The choice with the highest probability is: p = 12
```
</details>

To run this evaluation on one sample from AfriMMLU and investigate the output, you can run the following command:

```bash
python src/evaluate.py \
    --model-config-yaml configs/models/meta_llama_8b_instruct.yaml \
    --task-config-yaml configs/tasks/afrimmlu-direct.yaml \
    --eval.num-fewshot 0 \
    --eval.limit 1 \
    --eval.write-out
```
