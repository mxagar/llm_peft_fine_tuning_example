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

This site chronicles my observations in the fast-evolving landscape of data science,
covering topics related to AI/ML, computer vision, NLP, 3D, robotics... and more!
-->
<p align="center">
<img src="./assets/stochastic_parrot_dalle3.png" alt="A cheerful macaw parrot wearing sunglasses says 42." width="1000"/>
<small style="color:grey">Large Language Models (LLMs) have been called <a href="https://dl.acm.org/doi/10.1145/3442188.3445922">stochastic parrots</a> by some; in any case, they seem to be here to stay &mdash; and to be honest, I find them quite useful, if properly used. Let's see how they work. Image generated using <a href="https://openai.com/index/dall-e-3/">Dall-E 3</a>; prompt: <i> Wide, landscape cartoon illustration of a happy, confident red-blue-yellow macaw wearing black sunglasses, perched on a tree branch in a green forest, with a white comic speech bubble saying "42".</i>
</small>
</p>

The release of [ChatGPT](https://openai.com/blog/chatgpt) in November 2022 revolutionized our lives in the developed world. In a similar way as Google convinced us that the Internet is useful and we need their search engine or Apple presented the first actually usable smartphone that made the digital world ubiquitous, OpenAI came up with the next logical innovation: assitant chatbots based on Large Language Models (LLMs). Language models existed beforehand, but OpenAI's chat user interface and the emergent capabilities of their models coming from their humongous network and dataset sizes lead to the perfect killer app: the ever-ready genie that *seems* to know the answer to everything, confidently.

> It feels like *"ask ChatGPT"* has become the new *"google it"*.

Current LLMs are based on the **Transformer** architecture, introduced by Google in the seminal work [*Attention Is All You Need* (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762). Previous to that, [LSTMs or Long short-term memory networks (Hochreiter & Schmidhuber, 1997)](https://en.wikipedia.org/wiki/Long_short-term_memory) used to be state-of-the-art sequence models for Natural Language Processing (NLP). In fact, many of the concepts exploited by the Transformer were developed using LSTMs as the backbone, and one could argue that the LSTM still seems to be a more advanced model that the Transformer itself &mdash; if you'd like an example of an LSTM-based language modeller, you can check this [TV script generator of mine](https://mikelsagardia.io/blog/text-generation-rnn.html).

However, the Transformer presented some major *practical advantages* that enabled a paradigm shift:

- Its *self-attention* module made possible to convert sequential tasks into *parallelizable* ones. 
- Its uncomplicated, modular architecture made it easy to scale up and adapt to *many different tasks*.

Simultaneously, [Howard & Ruder (2018)](https://arxiv.org/abs/1801.06146) demonstrated that *transfer learning* worked not only in computer vision, but also for NLP: they showed that a language model pre-trained on a large corpus could be fine-tuned for smaller corpora and other downstream tasks.

And that's how the way to the current LLMs was paved. Nowadays, Transformer-based LLMs excel in *everything* NLP-related: text generation, summarization, question answering, code generation, translation, and so on.

## The Original Transformer and Its Siblings

Before 

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>



<p align="center">
<img src="./assets/llm_simplified.png" alt="LLM Simplified Architecture" width="1000"/>
<small style="color:grey">Caption.
</small>
</p>


## Deep Dive into the Architecture

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

## Where Do We Go from Here?


Links:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT](https://openai.com/index/language-unsupervised/)

Other important papers:

- GPT: decoder, generative model.
- BERT: encoder.
- Scaling laws.
- Emergent abilities.
- RLHF: Reinforcement Learning with Human Feedback.
- PEFT: Parameter-Efficient Fine-Tuning.
- RAG: Retrieval Augmented Generation.

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

Expert system for experts.

## Wrapping Up


