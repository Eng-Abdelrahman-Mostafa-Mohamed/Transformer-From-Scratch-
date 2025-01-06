# NLP Journy 
![diagram-export-1-6-2025-7_49_06-PM](https://github.com/user-attachments/assets/9006c395-ec32-4142-8a94-6123269e0b39)

* 1960s: Bernard Widrow and Marcian Hoff developed models called ADALINE.
* The modern deep learning era started in 2006.
* Before that, we worked with statistics, machine learning algorithms, and non-trainable algorithms.

### Why NLP and Deep Learning?

Why did we generate a whole branch of deep learning called NLP, which then upgraded to today's trends like LLMs or Vision-LLMs (same base structure)? Why didn't we use simple NN and simple encoding techniques like one-hot encoding or label encoding?

The goal is to train a model in context, not just on individual words. We want the model to understand the semantic meaning of a sentence as a whole, where each word depends on the previous ones.

### How NLP Relates to Computer Vision

Let's take an example from computer vision:

* We train a model for each pixel in an image.
* There's a relation between pixels as a whole to specify the image class (e.g., using CNNs with kernels).
* Similarly, we can use CNNs on encoded vectors (embeddings) in NLP. However, for images, the kernel moves in 2 dimensions (2D-CNN).
* Imagine embedded vectors for each word stored in an array. We want to use a kernel to extract data from a number of words (depending on kernel size). Here, we'd use 1D-CNN, but it's not ideal for complex NLP tasks.

Therefore, we need models that can "remember" weights (the knowledge of the model). These are called sequence models, which are the category of most LLMs and NLP models.

## Sequence Models

* RNN-LSTM-GRU (see [RNN-LSTM-GRU example](https://www.kaggle.com/code/abdelrahmanm2003/gen-rnn-model-ipynb))
* RNN-LSTM-GRU with Attention
* Transformer The Big Boss (The Core Model Of LLM - RAG)-4188-ab77-30a6c2953741.png

### What is Attention?

Attention is a technique that allows the model to understand how important each word is to the current context.


# Transformer

 - [](https://miro.medium.com/v2/resize:fit:989/0*Q7l2OWtPWiIZzT6T.png)

### Encoder

The encoder consists of several layers, each with the following components:

1. **Embedding Layer**
   - Converts input tokens into dense vectors of fixed size.
   ```
   Embedding(x) = W_e * x
   ```
   - Where:
     - `W_e` is the embedding matrix.
     - `x` is the input token.

2. **Positional Encoding**
   ![positional encoding](https://miro.medium.com/v2/resize:fit:1400/1*OB7gsRGz4Gm4qcKglpLSqQ.png)
   - Adds information about the position of each token in the sequence.
   ```
   PE_{(pos, 2i)} = sin(pos / 10000^(2i/d_model))
   PE_{(pos, 2i+1)} = cos(pos / 10000^(2i/d_model))
   ```
   - Where:
     - `pos` is the position.
     - `i` is the dimension.

4. **Multi-Head Attention**
   - Allows the model to focus on different parts of the input sequence.
   ```
   Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
   ```
   - Where:
     - `Q` (query), `K` (key), and `V` (value) are the input matrices.
     - `d_k` is the dimension of the key vectors.
   ```
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
   ```
   - Where:
     - `head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`

5. **Feed-Forward Network**
   - Applied to each position separately.
   ```
   FFN(x) = max(0, xW_1 + b_1) * W_2 + b_2
   ```
   - Where:
     - `W_1`, `W_2`, `b_1`, and `b_2` are learned parameters.

6. **Layer Normalization**
   - Stabilizes and accelerates the training process.
   ```
   LayerNorm(x) = (x - mu) / (sigma + epsilon) * gamma + beta
   ```
   - Where:
     - `mu` and `sigma` are the mean and standard deviation of the input.
     - `gamma` and `beta` are learned parameters.

7. **Residual Connection**
   - Helps in training deep networks.
   ```
   Output = LayerNorm(x + Sublayer(x))
   ```
   - Where `Sublayer(x)` can be multi-head attention or feed-forward network.

### Decoder

The decoder is similar to the encoder but with some differences:

1. **Masked Multi-Head Attention**
   - Prevents attending to future tokens.
   ```
   MaskedAttention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) * V
   ```

2. **Encoder-Decoder Attention**
   - Attends to the encoder's output.
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
   ```

3. **Feed-Forward Network**
   - Same as in the encoder.


#### We take about The architecture But How It actually Trained for Various Tasks (Translation(encoder_decoder_model) , classification(encoder only model),text generation "Trends now adays  😅 "(decoder only models)-->That's type of transformer is the top  used in advanced llm techniques like Rag - Agentic AI by integrate  multible llms together or llms+tools like ote saver ot code runner & evaluator)

