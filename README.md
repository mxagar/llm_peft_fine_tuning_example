# Applying Parameter-Efficient Fine-Tuning (PEFT) to a Large Language Model (LLM)

This example project shows how to fine-tune a Large Language Model using the PEFT library from HuggingFace.

The HuggingFace library [`transformers`](https://huggingface.co/docs/transformers/en/index) in combination with [`peft`](https://github.com/huggingface/peft) makes it very easy to fine-tune Large Language Models (LLMs) for our specific tasks.
This small project shows how to use those libraries end-to-end to perform a text classification task.

Specifically:

- We use the [`ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) dataset, which consists of 120k news texts, each of them with a label related to its associated topic: `'World', 'Sports', 'Business', 'Sci/Tech'`.
- The [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) model is fine-tuned for the news classification task. In the process, [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) is used to accelerate the fine-tuning thanks to the [`peft`](https://github.com/huggingface/peft) library.

The underlying LLM is abstracted and easily handled thanks to the [`transformers`](https://huggingface.co/docs/transformers/en/index) library; the user only needs to understand basic concepts such as

- Tokenization of text sequences
- Embedding vectors of tokens and associated dimensions
- The motivation and usage of the encoder & decoder modules in LLMs
- Task-specific heads, such as classification

![LLM Architecture Simplified](./assets/llm_simplified.png)

For a primer in those topics, you can visit

- [mxagar/generative_ai_udacity/01_Fundamentals_GenAI](https://github.com/mxagar/generative_ai_udacity/tree/main/01_Fundamentals_GenAI)
- [mxagar/nlp_with_transformers_nbs](https://github.com/mxagar/nlp_with_transformers_nbs)

## Setup

A recipe to set up a [conda](https://docs.conda.io/en/latest/) environment with the required dependencies:

```bash
# Create the necessary Python environment
conda env create -f conda.yaml
conda activate genai

# Compile and install all dependencies
pip-compile requirements.in
pip-sync requirements.txt

# If we need a new dependency,
# add it to requirements.in 
# And then:
pip-compile requirements.in
pip-sync requirements.txt
```

## Notebook

The notebook [`llm_peft.ipynb`](./llm_peft.ipynb) contains all the code and explanations necessary to perform the aforementioned fine-tuning.

## Interesting Links

- My personal notes on the O'Reilly book [Generative Deep Learning, 2nd Edition, by David Foster](https://github.com/mxagar/generative_ai_book)
- My personal notes on the O'Reilly book [Natural Language Processing with Transformers, by Lewis Tunstall, Leandro von Werra and Thomas Wolf (O'Reilly)](https://github.com/mxagar/nlp_with_transformers_nbs)
- My personal notes and guide for the [Generative AI Nanodegree from Udacity](https://github.com/mxagar/generative_ai_udacity/)
- [HuggingFace Guide: `mxagar/tool_guides/hugging_face`](https://github.com/mxagar/tool_guides/tree/master/hugging_face)
- [LangChain Guide: `mxagar/tool_guides/langchain`](https://github.com/mxagar/tool_guides/tree/master/langchain)
- [LLM Tools: `mxagar/tool_guides/llms`](https://github.com/mxagar/tool_guides/tree/master/llms)
- [NLP Guide: `mxagar/nlp_guide`](https://github.com/mxagar/nlp_guide)
- [Deep Learning Methods for CV and NLP: `mxagar/computer_vision_udacity/CVND_Advanced_CV_and_DL.md`](https://github.com/mxagar/computer_vision_udacity/blob/main/03_Advanced_CV_and_DL/CVND_Advanced_CV_and_DL.md)
- [Deep Learning Methods for NLP: `mxagar/deep_learning_udacity/DLND_RNNs.md`](https://github.com/mxagar/deep_learning_udacity/blob/main/04_RNN/DLND_RNNs.md)
