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
<small style="color:grey">Large Language Models (LLMs) have been called <a href="https://dl.acm.org/doi/10.1145/3442188.3445922">stochastic parrots</a> by some; in any case, they seem to be here to stay &mdash; and to be honest, I find them quite useful, if properly used. Image generated using <a href="https://openai.com/index/dall-e-3/">Dall-E 3</a>; prompt: <i> Wide, landscape cartoon illustration of a happy, confident red-blue-yellow macaw wearing black sunglasses, perched on a tree branch in a green forest, with a white comic speech bubble saying <a href="https://simple.wikipedia.org/wiki/42_(answer)">"42"</a>
.</i>
</small>
</p>


The release of [ChatGPT](https://openai.com/blog/chatgpt) in November 2022 revolutionized our lives in the developed world. In a similar way as Google convinced us that the Internet is useful and we need their search engine or Apple presented the first actually usable smartphone that made the digital world ubiquitous, OpenAI came up with the next logical innovation: assitant chatbots based on Large Language Models (LLMs). Language models existed beforehand, but OpenAI's chat user interface and the emergent capabilities of their models coming from their humongous network and dataset sizes lead to the perfect killer app: the ever-ready genie that *seems* to know the answer to everything, confidently.

> It feels like *"ask ChatGPT"* has become the new *"google it"*.

Current LLMs are based on the **Transformer** architecture, introduced by Google in the seminal work [*Attention Is All You Need* (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762). Previous to that, [LSTMs or Long short-term memory networks (Hochreiter & Schmidhuber, 1997)](https://en.wikipedia.org/wiki/Long_short-term_memory) used to be state-of-the-art sequence models for Natural Language Processing (NLP). In fact, many of the concepts exploited by the Transformer were developed using LSTMs as the backbone, and one could argue that the LSTM still seems to be a more advanced model that the Transformer itself &mdash; if you'd like an example of an LSTM-based language modeler, you can check this [TV script generator of mine](https://mikelsagardia.io/blog/text-generation-rnn.html).

However, the Transformer presented some major *practical advantages* that enabled a paradigm shift:

- Its *self-attention* module made possible to convert sequential tasks into *parallelizable* ones. 
- Its uncomplicated, modular architecture made it easy to scale up and adapt to *many different tasks*.

Simultaneously, [Howard & Ruder (2018)](https://arxiv.org/abs/1801.06146) demonstrated that *transfer learning* worked not only in computer vision, but also for NLP: they showed that a language model pre-trained on a large corpus could be fine-tuned for smaller corpora and other downstream tasks.

And that's how the way to the current LLMs was paved. Nowadays, Transformer-based LLMs excel in *everything* NLP-related: text generation, summarization, question answering, code generation, translation, and so on.

## The Original Transformer: Its Inputs, Components and Siblings

Before describing the components of the Transformer, we need to explain how text is represented for computers. In practice, text is converted into a **sequence of feature vectors** ${x_1, x_2, ...}$, each of dimension $m$ (the *embedding size* or *dimension*). This is done in the following steps:

1. **[Tokenization](https://en.wikipedia.org/wiki/Large_language_model#Tokenization)**: The text is split into discrete elements called *tokens*. Tokens are units with an identifiable meaning for the model and typically include words or sub-words, as well as punctuation and special symbols.
2. **Vocabulary construction**: A vocabulary containing all $n$ unique tokens is defined. It provides a mapping between each token string and a numerical identifier (token ID).
3. **[One-hot vectors](https://en.wikipedia.org/wiki/One-hot)**: Each token is mapped to its token ID. Conceptually, this corresponds to a one-hot vector of size $n$, although in practice models operate directly on token IDs. In a one-hot vector, all cells have the value $0$ except the cell which corresponds to the token ID of the represented word, which contains the value $1$.
4. **[Embedding vectors](https://en.wikipedia.org/wiki/Word_embedding)**: Token IDs (i.e., one-hot vectors) are mapped to dense embedding vectors using an embedding layer. This layer acts as a learnable lookup table (or equivalently, a linear projection of a one-hot vector), producing vectors of size $m$, with $m \ll n$. These embedding vectors are effectively arrays which contain floating point values. Typical reference values are $n \approx 100{,}000$ and $m \approx 500$.

<p align="center">
<img src="./assets/text_embeddings.png" alt="Text Embeddings" width="1000"/>
<small style="color:grey">A word/token can be represented as a one-hot vector (sparse) or as an embedding vector (dense). Embedding vectors allow to capture semantics in their directions and make possible a more efficient processing. Image by the author.
</small>
</p>

By the way, embeddings can be created for images, too, as I explain in [this post on diffusion models](https://mikelsagardia.io/blog/diffusion-for-developers.html). In general, they have some very nice properties:

- They build up a compact space, in contrast to the sparse one-hot vector space.
- They are continuous and differentiable.
- If the semantics is captured properly, words with close meaning are pointing to similar directions. As a consequence, we can perform arithmetics with them, such that algebraic operations (`+, -`) can be applied to words; for instance, the word `queen` is expected to be close to `king - man + woman`.

<p align="center">
<img src="./assets/text_image_embeddings.png" alt="Arithmetics with Text and Image Embeddings" width="1000"/>
<small style="color:grey">Embeddings can be computed for every modality (image, text, audio, video, etc.); we can even create multi-modal embedding spaces. If the embedding vectors capture meaning properly, similar concepts will have vectors to similar directions. As a consequence, we will be able to apply some algebraic operations on them. Image by the author.
</small>
</p>

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

The original Transformer was designed for language translation and it has two parts:

- The **encoder**, which converts the input sequence (e.g., a sentence in English) into hidden states or context.
- The **decoder**, which generates an output sequence (e.g., the translated sentence in Spanish) using as guidance some of the output hidden states of the encoder.

<p align="center">
<img src="./assets/llm_simplified.png" alt="LLM Simplified Architecture" width="1000"/>
<small style="color:grey">Simplified architecture of the original <a href="https://arxiv.org/abs/1706.03762">Transformer</a> designed for language translation. Highlighted: inputs (sentence in English), outputs (hidden states and translated sentence in Spanish), and main parts (the encoder and the decoder).
</small>
</p>

Using as reference the figure above, here's how the Transformer works:

- The encoder and the decoder are subdivided in `N` *encoder/decoder blocks* each; these blocks pass their hidden state outputs as inputs for the successive ones.

- The input of the first encoder block are the embedding vectors of the input text sequence. *Positional encodings* are added in the beginning to inject information about token order, since the self-attention layers inside the blocks (see next section) are position-agnostic. In the original paper, positional encoding vectors were $\mathbf{R} \rightarrow \mathbf{R}^n$ sinusoidal mappings: each unique scalar yielded a unique and different vector, thanks to systematically applying sinusoidal functions to the scalar. However, in practice learned positional embeddings are often used instead.

- For the translation task the encoder input contains the representation of the full original text sequence; meanwhile, the decoder produces the output sequence one by one, but it always has the the full and final encoder hidden state (the context).

- The *decoder blocks* work in a similar way as the *encoder blocks*; the last *decoder block* produces the final set of hidden states, which are mapped to output token probabilities using a linear layer followed by a softmax function (i.e., we have a classification head over the vocabulary).

Soon after the publication of the original encoder-decoder Transformer designed for the language translation task, two related, important Transformers were introduced:

- [**BERT**: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805), which is an implementation of the **encoder-only** part of the original Transformer. 
- [**GPT**: Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), an implementation of the **decode-only** part of the original Transformer.

BERT-like *encoder-only* transformers are commonly used to generate *feature vectors* $x$ of texts which can be used in downstream applications such as text or token/word classification. If the Encoder is trained separately, the sequence text is shown to the architecture with a masked token which needs to be predicted. This scheme is called *masked language modeling*.

GPT-like *decoder-only* transformers are commonly used as *generative models* to predict the next token in a sequence, given all the previous tokens (i.e., the context, which includes the prompt). During training, the model is shown sequences of text and learns to predict each token based on the preceding ones.

The full *encoder-decoder* architecture is not as common as the other two currently, but it is used in some specific models for text-to-text tasks, such as summarization and translation. Examples include [T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683) and [BART (Lewis et al., 2019)](https://arxiv.org/pdf/1910.13461).

## Deep Dive into the Transformer Architecture

So far, we've seen the big picture of the Transformer architecture and its subtypes (encoder-decoder, encoder-only, decoder-only).

> But what's inside those encoder and decoder blocks? Just Attention, normalization, and linear mappings. Let's see them in detail.

<p align="center">
<img src="./assets/transformer_annotated.png" alt="Transformer Architecture, Annotated" width="1000"/>
<small style="color:grey">The Transformer architecture with all its components. Image from the orinal paper by <a href="https://arxiv.org/abs/1706.03762">Vaswani et al. (2017)</a>, modified by the author.
</small>
</p>

As we can see in the figure above, each of the `N` encoder and decoder blocks are composed of the following sub-components:

- **Multi-Head Self-Attention modules**: The core component of the Transformer. It allows the model to focus on different parts of the input sequence when processing each token. Multiple attention heads enable the model to capture various relationships and dependencies in the data. More on this below :wink:
- **Skip connections, Add & Norm**: These are [residual (skip) connections](https://arxiv.org/abs/1512.03385) followed by [layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization). Residual connections help to avoid vanishing gradients in deep networks by allowing gradients to flow directly through the skip connections. Normalizing the inputs across the features dimension stabilizes and accelerates training.
- **Feed-Forward Neural Network** (FFNN, i.e., several concatenated linear mappings): A fully connected feed-forward network applied independently to each position. It consists of two linear transformations with a [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) activation in between, allowing the model to learn complex representations.

The key contribution of the Transformer architecture is the **Self-Attention** mechanism. Attention was introduced by [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) and it allows the model to weigh the importance of different tokens in the input sequence when processing each token. In practice for the Transformer, similarities of the tokens in the sequence are computed simultaneously (i.e., dot product) and used to weight and sum the embeddings in successive steps.

We can see there are different types of attention modules in the Transformer:

- Self-Attention in the encoder blocks: Each token attends to the similarities of *all* tokens in the input sequence. It's called self-attention, because the similarities of the input tokens only are used, i.e., without any interaction with the decoder. For more information, keep reading below.
- Masked Self-Attention in the decoder blocks: Each token attends to *all previous* tokens in the output sequence (masked to prevent attending to future tokens).
- Encoder-Decoder Cross-Attention in the decoder blocks: Each token in the output sequence attends to *all tokens in the encoder-input sequence*. In other words, all final hidden states from the encoder are used in the attention computation.

Additionally, each attention module is implemented as a **Multi-Head Attention** mechanism. This means that multiple attention heads are used in parallel. The following figure shows brief overview of how this works.

<p align="center">
<img src="./assets/llm_attention_architecture.png" alt="LLM Attention Architecture" width="1000"/>
<small style="color:grey">The LLM (Self-)Attention module, annotated. Image by the author.
</small>
</p>

The **Self-Attention Head** is the core implementation of the attention mechanism in the Transformer. Each multi-head attention module contains $n$ self-attention heads, which operate in parallel. The input embedding sequence $Z$ passed to each of these $n$ self-attention heads, where the following occurs:

- We transform the original embeddings $Z$ into $Q$ (query), $K$ (key), and $V$ (value). The transformation is performed by linear/dense layers ($W_Q$, $W_K$, $W_V$), which consist of the learned weights. These *query*, *key*, and *value* variables come from classical [information retrieval](https://en.wikipedia.org/wiki/Information_retrieval); as described in [NLP with Transformers (Tunstall et al., 2022)](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/), using the analogy to a recipe they can be interpreted as follows:
    - $Q$, *queries*: ingredients in the recipe.
    - $K$, *keys*: the shelf-labels in the supermarket.
    - $V$, *values*: the items in the shelf.
- $Q$ and $K$ are used to compute a similarity score between token embedding against token embedding (*self* dot-product), and then we multiply the similarity scores to the values $V$, so the relevant information is amplified. This can be expressed mathematically with the popular and simple *attention* formula:
  $$Y = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V,$$
  where
    - $Y$ are the *contextualized embeddings*,
    - and $d_k$ is the dimension of the key vectors (used for scaling), which is the same as the embedding size divided by the number of heads (head dimension).

Then, these $Y_1, ..., Y_n$ contextualized embeddings are concatenated and linearly transformed to yield the final output of the multi-head attention module. The output of the first multi-head self-attention module is the input of the next one, and so on, until all $N$ blocks process embedding sequences. Note that the output embeddings from each encoder block have the same size as the input embeddings, so the encoder block stack has the function of *transforming* those embeddings with the attention mechanism.

> I hope now it's clear the title of the Transformer paper *Attention Is All You Need*: It turns out that successively focusing and transforming the embeddings via the attention mechanism produces the magic in the LLMs.

Finally, let's see some typical size values, for reference:

- Embedding size: 768, 1024, ..., 2048.
- Sequence length (context, number of tokens): 128, 256, ..., 8192.
- Number of layers/blocks, $N$: 12, 24, 36, 48.
- Number of attention heads, $n$: 12, 16, 20, 32.
- Head dimension: typically, embedding size divided by number of heads.
- Feed-Forward Network (FFN) inner dimension: 2048, 4096, ..., 10240.
- Vocabulary size, $m$: 30,000; 50,000; 100,000; 200,000.
- Total number of parameters: from 110 million (e.g., BERT-base) to 175 billion (e.g., GPT-3), and much more!

<div style="height: 20px;"></div>
<div align="center" style="border: 1px solid #e4f312ff; background-color: #fcd361b9; padding: 1em; border-radius: 6px;">
<strong>
If you are interested on an implementation of the Transformer, you can check <a href="https://github.com/mxagar/nlp_with_transformers_nbs/blob/main/03_transformer-anatomy.ipynb">this notebook</a>, where I modified the code from the official repository of the book <a href="https://www.oreilly.com/library/view/natural-language-processing/9781098136789/">NLP with Transformers (Tunstall et al., 2022)</a>. In the same <a href="github.com/mxagar/nlp_with_transformers_nbs/">repository</a>, you'll find many other notebooks related to NLP with Transformers.
</strong>
</div>
<div style="height: 30px;"></div>

## Using the Transformer Outputs

There are many ways in which the outputs of the Transformer can be used, depending on the task and the architecture (some of these ways were mentioned above already):

- Encoder-decoder models (e.g., [T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683)) have been used for *text-to-text* tasks, such as *translation* and *summarization*. However, big enough decoder-only models (e.g., [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)) have shown remarkable performance in these tasks, too, and have become more popular nowadays.
- Encoder-only models (e.g., [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)) are commonly used to generate *feature vectors* of texts, which can be used in downstream applications such as text or token/word classification, or even regression. We just need to attach the proper mapping head to the output of the encoder (e.g., a linear layer for classification) and fine-tune the model on the specific task.
- Decoder-only models (e.g., [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)) are commonly used as *generative models* to predict the next token in a sequence, given all the previous tokens (i.e., the context, which includes the prompt).

Probably, the most common way to interact with LLMs for the layman user is the latter: decoder-only *generative models*. As mentioned, these models generate one word/token at a time, so we introduce their ouputs as inputs for successive generations (hence, they are called *autoregressive*). In that scheme, we need to consider the following questions:

1. *Which tokens are considered as candidates every generation?* (token sampling)
2. *Which is the strategy used to select and chain the tokens?* (token search during decoding)

Recall that the output of the generative model is an array of probabilities, specifically, a float value $p \in [0,1]$ for each item in the vocabulary set $V$. A naive approach would be to

1. consider all token probabilities as candidates $\{p_1, p_2, ...\}$ (full distribution sampling),
2. and select the token with the highest probability at each generation step: $\mathrm{token} = V(\mathrm{argmax}\{p_1, p_2, ...\})$ (greedy search decoding).

However, such a naive approach can lead to repetitive and dull text generation, as described by [Holtzman et al. (2019)](https://arxiv.org/abs/1904.09751). To mitigate this issue, these parameters and strategies are often used:

- Temperature: we apply the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function to the probabilities using the inverse of a $T$ *temperature* variable as the exponent ([Boltzman distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution)): $p_i' = \exp(\frac{p_i}{T}) / \sum_j \exp(\frac{p_j}{T})$. That changes the $p$ values as follows:
  - $T = 1$: no change, same as in the original output.
  - $T < 1$: small $p$-s become smaller, larger $p$-s become larger; that means we get a more peaked distribution, i.e., less creativity and more coherence, because the most likely words are going to be chosen.
  - $T > 1$: small $p$-s become bigger, larger $p$-s become smaller; that yields a more homogeneous distribution, which leads to more creativity and diversity, because any word/token could be chosen.
- Top-$k$ and top-$p$: instead of considering all tokens each with their $p$ (with or without $T$), we reduce it to the $k$ most likely ones and select from them using the distribution we have; similarly, with a top-$p$, we can select the first tokens that cumulate up to a certain $p$-threshold and choose from them.
- Beam search decoding (as oposed to greedy search): we select a number of beams $b$ and keep track of the most probable next tokens building a tree of options. The most likely paths/beams are chosen, ranking the beams with their summed log probabilities. The higher the number of beams, the better the quality, but the computational effort explodes. Beam search sometimes suffers from repetitive generation; one way to avoid that is using n-gram penalty, i.e., we penalize the repetition of n-grams. This is commonly used in summarization and machine translation.

## Additional Relevant Concepts

My goal with this post was to explain in plain but still technical words how LLMs work internally. In that sense, I guess I have already given the best I could and I should finish the text. However, there are some additional details that probably fit nicely as appendices here. Thus, I have decided to include them with a brief description and some references, for the readers who optionally want to go deeper into the topic.

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>


[InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)

[Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)

[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)

[LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)

[Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

**Context Size** &mdash; This refers to the maximum number of words/tokens that the model can consider as input at once, i.e., the input sequence legth or `seq_len`. If we look at the attention mechanism figure above, we will see that the learned weight matrices are independent of the context size; however, the attention computation itself scales quadratically with sequence length due to the $QK^T$ operation. This is a major bottleneck in terms of memory and speed, and it's the main reason why the initial LLMs had a fixed and shorter context size ($512$ - $4,096$ tokens). In recent years, the research community has explored new methods to alleviate that limitation, introducing techniques such as [sparse attention](https://arxiv.org/abs/2004.05150), [linearized attention](https://arxiv.org/abs/2006.16236), [low-rank approximations](https://arxiv.org/abs/2006.04768), and other mathematical/architectural/system tricks. These enable larger context sizes (up to $1,000,000$ tokens in the case of [Gemini Pro](https://gemini.google.com/app)).

**Distillation and Quantization** &mdash; As their name indicates, Large Language Models are *large*, and that makes them difficult to deploy in production environments. Two techniques to overcome that are *distillation* and *quantization*. When we distill a model, we train a smaller student model to mimic the behavior of a larger, slower but better prforming teacher (i.e., the original LLM). This achieved, among others, by using the teacher's output probabilities as soft labels for the outputs of the training the student. A notable example of distillation is [DistilBERT (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108), which achieves around 97% of BERT's performance, but with 40% less memory and 60% faster inference. On the other hand, *quantization* consists in representing the weights with lower precision, i.e., `float32 -> int8` ($32/8 = 4$ times smaller models). The models not only become smaller, but the operations can be done faster (even 100x faster), and the accuracy is sometimes similar.

**Emergent Abilities** &mdash; As described by [Wei et al. (2022)](https://arxiv.org/abs/2206.07682), *"emergent abilities are those that are not present in smaller models, but appear in larger ones"*. In other words, they are capabilities that arise, but which were not explicitly trained. This often referred as *zero-shot* or *few-shot* learning, because the model can perform tasks without any or with very few examples, as demonstrated by [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165), and they start to appear in the 10-100 billion parameter range (GPT-3 had 175 billion parameters). Examples of emergent abilities include arithmetic, commonsense reasoning, and even some forms of creativity. 

**Scaling Laws** &mdash; Kaplan et al. published in 2020 the interesting paper [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), which describes how the performance of language models scales. They discovered that there is a power-law relationship between the model's performance measured in terms of loss $L$, the required compute $C$, the dataset size $D$ and the model size $N$ (number of parameters): $L(X) \sim X^{-\alpha}$, with $X \in \{N, C, D\}$ and $\alpha \in [0.05, 0.1]$. In other words, when model size $N$, dataset size $D$, or training compute $C$ are scaled independently (and are not bottlenecks), the training loss $L$ decreases approximately as a power law of each quantity. In that sense, we can use these scaling laws to extrapolate model performance without building them, but theoretically! Similarly, for a fixed compute budget, there is an optimal trade-off between model size and dataset size. These insights led to the development of more efficient training strategies and architectures, such as the ones explored in the [Chinchilla study (Hoffman et al., 2022)](https://arxiv.org/abs/2203.15556), which suggest that smaller models trained on more data can achieve better performance than larger models trained on less data. Finally, note that training compute is roughly proportional to $6 \times N \times D$, while inference compute scales linearly with model size and generated sequence length. For reference, the following table shows some size values:

| Quantity                                      | Low                       | Medium                    | Large                                      |
| --------------------------------------------- | ------------------------- | ------------------------- | ------------------------------------------ |
| **Dataset (tokens)**             | ~10⁹ tokens *(BERT-base)* | ~10¹¹ tokens *(LLaMA-7B)* | ~10¹²–10¹³ tokens *(GPT-4 / Gemini-class)* |
| **Model (parameters)**           | ~10⁸–10⁹ *(DistilBERT)*   | ~10¹⁰ *(LLaMA-13B)*       | ~10¹¹–10¹² *(GPT-3 / MoE models)*          |
| **Training Compute (FLOPs)**          | ~10²⁰–10²¹ *(BERT-base)*  | ~10²²–10²³ *(LLaMA-7B)*   | ~10²⁴–10²⁵ *(GPT-3 / frontier LLMs)*       |
| **Inference Compute (FLOPs / token)** | ~10⁸–10⁹ *(DistilBERT)*   | ~10¹⁰–10¹¹ *(LLaMA-13B)*  | ~10¹¹–10¹² *(GPT-3-class)*                 |



**RLHF: Reinforcement Learning with Human Feedback** &mdash;

**Mixture of Experts** &mdash;

**Reasoning Models** &mdash;

**PEFT: Parameter-Efficient Fine-Tuning** &mdash;

**RAG: Retrieval Augmented Generation** &mdash;

**Agents** &mdash;

## Wrapping Up

Summary

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

Expert system for experts.
Productivity
Conciousness
World model

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

Links:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**
- [A minimal PyTorch re-implementation of the OpenAI GPT (Andrej Karpathy)](https://github.com/karpathy/minGPT)
