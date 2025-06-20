# Applying Parameter-Efficient Fine-Tuning (PEFT) to a Large Language Model (LLM)

This example project shows how to fine-tune a Large Language Model using the PEFT library from HuggingFace.

## Notes

`tweet_eval` is a series of datasets; particularly, we will use the `emotion` subset, which assigns a class label ` 0: anger, 1: joy, 2: optimism, 3: sadness` to a each of the 3257 tweets in the training split.
Given the small dataset, it makes sense to use a small model as DistilBERT.
DistilBERT is a distillation of BERT one of the first encoder-only transformer models, trained basically on Masked Language Modeling (MLM) -- predicting a masked word.
An alternative for larger datasets could be RoBERTa, which was trained roughly on 10x more data than BERT, and has roughly double the parameters than DistilBERT.
We could use other models, e.g., generative decoder transformers like GPT2, although in general RoBERTa seems to have better performance for classification tasks.
GPT-2 is similar in size to RoBERTa.

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
