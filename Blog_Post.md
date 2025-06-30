# Blog Post: Applying Parameter-Efficient Fine-Tuning (PEFT) to a Large Language Model (LLM)

TBD.

:construction:

`tweet_eval` is a series of datasets; particularly, we will use the `emotion` subset, which assigns a class label ` 0: anger, 1: joy, 2: optimism, 3: sadness` to a each of the 3257 tweets in the training split.
Given the small dataset, it makes sense to use a small model as DistilBERT.
DistilBERT is a distillation of BERT one of the first encoder-only transformer models, trained basically on Masked Language Modeling (MLM) -- predicting a masked word.
An alternative for larger datasets could be RoBERTa, which was trained roughly on 10x more data than BERT, and has roughly double the parameters than DistilBERT.
We could use other models, e.g., generative decoder transformers like GPT2, although in general RoBERTa seems to have better performance for classification tasks.
GPT-2 is similar in size to RoBERTa.