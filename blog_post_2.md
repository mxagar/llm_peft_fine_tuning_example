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
<small style="color:grey">Two <a href="https://dl.acm.org/doi/10.1145/3442188.3445922">stochastic parrots</a> dressed up like Star Wars and Star Trek characters; same parrot, different costumes and roles. Image generated using <a href="https://openai.com/index/dall-e-3/">Dall-E 3</a>; prompt: <i> Wide landscape cartoon illustration of two red-blue-yellow macaws with sunglasses on tree branches in a bright green forest. Left parrot dressed as a Jedi with robe and blue lightsaber, right parrot dressed as a classic Star Trek Vulcan officer in a gold uniform. Bold, vibrant vector style.</i>
</small>
</p>

In my [previous post](https://mikelsagardia.io/posts/) I explained how LLMs are built, and how they work. In this post, I will try to explain how to adapt LLMs easily to specific *tasks* and *domains* using [HuggingFace's `peft` library](https://github.com/huggingface/peft). As explained in the official site, [PEFT or Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/en/index) is a family of techniques that

> only fine-tune a small number of (extra) model parameters &mdash; significantly decreasing computational and storage costs &mdash; while yielding performance comparable to a fully fine-tuned model. This makes it more accessible to train and store large language models (LLMs) on consumer hardware.

In summary, I cover the following topics in this post:

- A
- B
- C

Let's start!

<div style="height: 20px;"></div>
<div align="center" style="border: 1px solid #e4f312ff; background-color: #fcd361b9; padding: 1em; border-radius: 6px;">
<strong>
You can find this post's accompanying code in <a href="https://github.com/mxagar/diffusion-examples/ddpm">this GitHub repository</a>.
</strong>
</div>
<div style="height: 30px;"></div>

## Why and How Should We Adapt LLMs?

First of all, we should define some terminology:

- A *Task*: a specific problem we want to solve. The task is usually defined by the *input* and the *output* formats. Typically, LLMs are trained on the general task of *language modeling*: predicting the next word/token given an input sequence (i.e., the context); as such, they are able to generate coherent text related to the input. However, we can change their output layers (also known as *heads*) to perform other tasks, such as *text classification* (e.g., *sentiment analysis* and *topic classification*), *token classification* (e.g., *named entity recognition* or NER), etc.
- A *Domain*: the specific area or context to which the training texts belong and in which the task needs to be performed. Typically, LLMs are trained on a wide variety of texts from the Internet, which makes them generalists. However, we may want to adapt them to specific domains, such as *medicine*, *finance*, *legal*, etc. The more niche the domain, the more we may need to adapt the LLM to it to learn style, jargon, and specific knowledge.

This *task* and *domain* adaptation, although it is named as *fine-tuning* for LLMs, is known as *transfer learning* in the context of computer vision. It was [Howard and Ruder (2017)](https://arxiv.org/abs/1801.06146) who showed that a language model trained on a large corpus can be re-adapted for smaller corpora and other downstream tasks.

One common approach in the [PEFT](https://huggingface.co/docs/peft/en/index) library is the [Low-Rank Adaptation (or LoRA, introduced by Hu et al., 2021)](https://arxiv.org/abs/2106.09685), which I cover in more detail in the next section. In a nutshell: LoRA freezes the pre-trained weight matrices $W$ and adds to them new matrices $dW$, which are the ones that are trained. These $dW$ matrices are factored as the multiplication of two low-rank matrices; that trick reduces trainable parameters by orders of magnitude and maintains or matches full fine-tuning performance on many benchmarks.

There are other ways to adapt LLMs which I won't cover here, such as:

- [RLHF (Reinforcement Learning with Human Feedback)](https://arxiv.org/abs/2203.02155): This technique was used to align the initial ChatGPT model (GPT 3.5) with human preferences. Initially, human annotators ranked outputs of a GPT model. Then, these annotations were used to train a reward model (RM) to automatically predict the output score. And finally, the GPT model (*policy*) was trained using the [Proximal Policy Optimization (PPO) algorithm](https://en.wikipedia.org/wiki/Proximal_policy_optimization), based on the conversation history (*state*) and the outputs it produced (*actions*), and using the reward model (*reward*) as the evaluator.
- [RAG (Retrieval Augmented Generation)](https://arxiv.org/abs/2005.11401): This method consists in outsourcing the domain-specific memory of LLMs. In an offline ingestion phase, the knowledge is chunked and indexed, often as embedding vectors. In the real-time generation phase, the user asks a question, which is encoded and used to retrieve the most similar indexed chunks; then, the LLM is prompted to answer the question by using the found similar chunks, i.e., the retrieved data is injected in the query. RAGs reduce hallucinations and have been extensively implemented recently.

In my experience, usually PEFT/LoRA and RAG are the most used techniques and they can be used in combination:

- PEFT/LoRA makes sense when we need to approach a task different than *language modeling* (i.e., next token prediction), or when we have a very specific domain, such as *medicine* or *finance*, which is not well represented in the general training data of the LLM.
- RAG is more useful when we have a task that can be solved by retrieving specific information, such as *question answering* or *summarization*, and when we have a large amount of domain-specific data that changes constantly. Most chatbots that are used in production for customer support, for instance, are RAG-based.

### How Does PEFT/LoRA Work?

When we apply Low-Rank Adaptation (LoRA), we basically decompose a weight matrix into a multiplication of low-rank matrices that have fewer parameters.

Let's consider a pre-trained weight matrix $W$; instead of changing it directly, we add to it a weight offset $dW$ as follows:

$$W = W + dW,$$

where 

- $W$ represents a weight matrix of shape $(d, f)$
- and $dW$ is a weight offset to be learned, of shape $(d, f)$.

However, we do not operate directly with the weight offset $dW$; instead, we factor it as the multiplication of two low-rank matrices:

$$dW = A \cdot B,$$

where

- $A$ is of shape $(d, r)$,
- $B$ is of shape $(r, f)$,
- and $r << d, f$.

The key idea is that during training we freeze $W$ while we learn $dW$, but instead of learning the full sized $dW$, we learn the much smaller $A$ and $B$. The forward pass of the model is modified as follows:

$$y = x \cdot W = x \cdot (W + dW) = x \cdot (W + A \cdot B).$$

The proportion of weights in $dW$ as compared to $W$ is the following:

- Weights of $W$: $d \cdot f$
- Weights of $A$ and $B$: $r \cdot (d + f)$
- Proportion: $r\cdot\frac{d + f}{d \cdot f}$

Note that the number of trainable parameters is reduced by controlling the rank value $r$; for instance, if we set $r=4$, we can reduce the number of trainable parameters by more than `100x` for a weight matrix of size $(4096, 4096)$.

LoRA is not applied to all weight matrices, but usually the library (`peft`) decides where to apply it; e.g.: projection matrices $Q$ and $V$ in attention blocks, MLP layers, etc. And, after training, we can merge $W + dW$, so there is no latency added!

If you want to learn more about PEFT and LoRA, I recommend checking the following resources:

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Hugging Face LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

## Implementation

Thanks to the `peft` library, applying PEFT/LoRA to an LLM is very easy. The [Github repository](#) contains the Jupyter Notebook [`llm_peft.ipynb`](#) in which an example is provided.

In the example, I use DistilBERT, which is a smaller version of BERT that has been distilled to reduce its size and computational requirements while maintaining good performance.

DistilBERT is a distillation of BERT one of the first encoder-only transformer models, trained basically on Masked Language Modeling (MLM) -- predicting a masked word.
An alternative for larger datasets could be RoBERTa, which was trained roughly on 10x more data than BERT, and has roughly double the parameters than DistilBERT.

We could use other models, e.g., generative decoder transformers like GPT2, although in general RoBERTa seems to have better performance for classification tasks.
GPT-2 is similar in size to RoBERTa.

Finally, if you are interested, consider checking these additional resources:

- A
- B
- C

## Results

## Conclusion




