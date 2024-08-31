# LLM Evaluation for African Languages

## Description

This project is aimed at creating a standard evaluation library and scripts for assessing the performance of language models on tasks related to African languages. This project builds upon the [lm-evaluation-harness by EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness), extending its capabilities to focus on African language tasks. Some of the tasks contained in this leaderboard have already been built into the harness (e.g. IrokoBench Tasks) but we make some slight modifications to some tasks such as 

- Improving the formatting of the prompts because language models have been shown to be sensitive to the formatting of the prompts ([Sclar et al.](https://arxiv.org/pdf/2310.11324)). Specifically, we strip the prompt of excess whitespace and whitelines and next-line characters.

- Making the leaderboard reproducible by fixing the seeds for generating the fewshot examples needed for the tasks.

- Explore a scenario where the system prompts / prompt templates are written in the language being evaluated other than English.

## Tasks

### Multiple Choice Tasks

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

### How does LM-Harness evaluate multiple choice tasks?