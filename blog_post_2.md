# Applying Parameter-Efficient Fine-Tuning (PEFT) to a Large Language Model (LLM)

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
<img src="./assets/scifi_parrots_dalle3.png" alt="Two cheerful macaw parrots dressed in Star Wars and Star Trek outfits." width="1000"/>
<small style="color:grey">Two <a href="https://dl.acm.org/doi/10.1145/3442188.3445922">stochastic parrots</a> dressed up like Star Wars and Star Trek characters; same parrot, different costumes. Image generated using <a href="https://openai.com/index/dall-e-3/">Dall-E 3</a>; prompt: <i> Wide landscape cartoon illustration of two red-blue-yellow macaws with sunglasses on tree branches in a bright green forest. Left parrot dressed as a Jedi with robe and blue lightsaber, right parrot dressed as a classic Star Trek officer in a gold uniform. Bold, vibrant vector style.</i>
</small>
</p>


ULMFiT (Howard, 2017): a language model trained on a large corpus can be re-adapted for smaller corpora and other downstream tasks. Transfer learning was validated also for NLP; until then, only worked in CV.

DistilBERT is a distillation of BERT one of the first encoder-only transformer models, trained basically on Masked Language Modeling (MLM) -- predicting a masked word.
An alternative for larger datasets could be RoBERTa, which was trained roughly on 10x more data than BERT, and has roughly double the parameters than DistilBERT.

We could use other models, e.g., generative decoder transformers like GPT2, although in general RoBERTa seems to have better performance for classification tasks.
GPT-2 is similar in size to RoBERTa.



With Low-Rank Adaptation, we basically decompose a weights matrix into a multiplication of low rank matrices:

    W = W + dW, where
    W: weight matrix (d, d)
    dW: weight offset to be learned (d,d)
    dW = A*B, where
    A is (d, r)
    and B (r, f)
    and r << d

The idea is that we freeze W while we learn dW, but instead of learning the full sized dW, we learn the much smaller A and B; the size of these low-rank matrices is controlled by r.

    y = x * W
    y = x * (W + dW) = x * (W + A*B), where
    x*W is frozen
    x*A*B is trainable

So, if W is (d, d) and A and B have rank r, the proportion of weights in dW as compared to W is:

    Weights W: d^2
    Weights A and B: 2 * (r*d)
    Proportion: 2*r/d

Additional notes:

- LoRA is not applied to all weight matrices, but the library (peft) decides where to apply it; e.g.: projection matrices Q and V in attention blocks, MLP layers, etc.
- We dramatically reduce the number of parameters by controlling r.
- LoRA can be used in combination with other methods.
- Performance is comparable to fully fine-tuned models!
- After training, we can merge W + dW, so there is no latency added!

More information:

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Hugging Face LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
