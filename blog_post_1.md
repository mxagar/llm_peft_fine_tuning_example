# How Are Large Language Models (LLMs) Built?

<!--

Excalidraw:

```bash
# Log in/out to Docker Hub
docker logout
docker login

# Pull the official image (first time)
docker pull excalidraw/excalidraw

# Start app
docker run --rm -dit --name excalidraw -p 5001:80 excalidraw/excalidraw:latest
# Open browser at http://localhost:5001

# Stop
docker stop excalidraw
docker rm excalidraw
docker ps
```

Blog Post 1: How Are Large Language Models (LLMs) Built?
Subtitle: A Conceptual Guide for Developers

Blog Post 2: Applying Parameter-Efficient Fine-Tuning (PEFT) to a Large Language Model (LLM)
Subtitle: When We Need to Adapt LLMs to Specific Tasks and Domains

Blog Post 3: Retrieval Augmented Generation (RAG) with LLMs: Some Blueprints
Subtitle: How to Use External Knowledge Bases to Enhance LLM Responses


This site chronicles my observations in the fast-evolving landscape of data science.
You'll find my explorations of AI/ML topics spanning computer vision, NLP, 3D, robotics... and more.

-->


<p align="center">
<img src="./assets/stochastic_parrot_dalle3.png" alt="A cheerful macaw parrot wearing sunglasses says 42." width="1000"/>
Large Language Models (LLMs) have been called <small style="color:grey"><a href="https://dl.acm.org/doi/10.1145/3442188.3445922">stochastic parrots</a> by some; in any case, they seem to be here to stay &mdash; and to be honest, I find them quite useful, if properly used. Let's see how they work. Image generated using <a href="https://openai.com/index/dall-e-3/">Dall-E 3</a>; prompt: <i> Wide, landscape cartoon illustration of a happy, confident red-blue-yellow macaw wearing black sunglasses, perched on a tree branch in a green forest, with a white comic speech bubble saying "42".</i>
</small>
</p>

The release of [ChatGPT](https://openai.com/blog/chatgpt) in November 2022 revolutionized our lives in the developed world. In a similar way as Google convinced us we need the Internet and their search engine or Apple presented the first actually usable smartphone that made the digital world ubiquitous, OpenAI came up with the next logical innovation: assitant chatbots based on Large Language Models (LLMs). Language models existed beforehand, but OpenAI's chat user interface and the emergent capabilities of their models coming from their humongous network and dataset sizes lead to the perfect killer app: the ever-ready genie that *seems* to know the answer to everything, confidently.

> It feels like *"ask ChatGPT"* has become the new *"google it"*.





[An Infinite Text Generator](https://mikelsagardia.io/blog/text-generation-rnn.html)

<p align="center">
<img src="./assets/llm_simplified.png" alt="LLM Simplified Architecture" width="1000"/>
<small style="color:grey">Caption.
</small>
</p>


<p align="center">
<img src="./assets/llm_attention_architecture.png" alt="LLM Attention Architecture" width="1000"/>
<small style="color:grey">Caption.
</small>
</p>


<p align="center">
<img src="./assets/transformer_annotated.png" alt="Transformer Architecture, Annotated" width="1000"/>
<small style="color:grey">Caption.
</small>
</p>


DistilBERT is a distillation of BERT one of the first encoder-only transformer models, trained basically on Masked Language Modeling (MLM) -- predicting a masked word.
An alternative for larger datasets could be RoBERTa, which was trained roughly on 10x more data than BERT, and has roughly double the parameters than DistilBERT.

We could use other models, e.g., generative decoder transformers like GPT2, although in general RoBERTa seems to have better performance for classification tasks.
GPT-2 is similar in size to RoBERTa.

Links:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT](https://openai.com/index/language-unsupervised/)

