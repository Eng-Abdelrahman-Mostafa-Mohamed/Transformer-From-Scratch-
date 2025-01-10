# NLP Journy 
![NLP Journy](https://github.com/Eng-Abdelrahman-Mostafa-Mohamed/Transformer-From-Scratch-/blob/Eng-Abdelrahman-Mostafa-Mohamed/NLp_%20journy.png)

* 1960s: Bernard Widrow and Marcian Hoff developed models called ADALINE.
* The modern deep learning era started in 2006.
* Before that, we worked with statistics, machine learning algorithms, and non-trainable algorithms.

### Why NLP and Deep Learning?

Why did we generate a whole branch of deep learning called NLP, which then upgraded to today's trends like LLMs or Vision-LLMs (same base structure)? Why didn't we use simple NN and simple encoding techniques like one-hot encoding or label encoding?

The goal is to train a model in context, not just on individual words. We want the model to understand the semantic meaning of a sentence as a whole, where each word depends on the previous ones.

### How DL or ML model trained or understand  the text?
####      as the models doesnt understand any thing nan(not a number ) --> we have multible encoding techniques devidied into 2 Main  chatagories .
   - Non Trainable    
      - encoding by index only  [problem is OOV ( out of vocabulary and Memory cosuming)]
      - encoding by index and frequancy 
   - Trainable encoding 
      - embeddings 

- Non Trainable encoding 
   - one hot encoding (sparse) -> put the index of word 1 and others zero 
   - Bag of word =   one hot encoding matrix * frequancy of each word matrix 
   - TF=IDF (Term frequency-inverse document frequency)-->  you could see My notebook how to build it from scratch
    [TF-Idf From Scratch](https://www.kaggle.com/code/abdelrahmanm2003/tfidf-with-llm-from-scratch)
      ![alt text](image-1.png)
      - TF: 
         - how this word is importand for comparing between 2 documents
         - work in each document separatly.

      - IDF:  
         - how this word is not important for comparing between 2 douments (same words and its frequancy nearly same in all documents ).
         - word with all documents  

- Trainable encoding 
   - word embeddings 
      -there is many examples of word embeddings like
         - Fast Text 
         - word-To-vec
         - Glove  
      - What is word embedding (Word To Vec) ?
         - is technique to encode word of sentense with semantic Knowledge 
            ![alt text](image-5.png)
            - as we see we represent each word in N-space Worled -->(the distance between words that have same semantic meaning is small and in releated word is the big .)
         - we may apply mathimatical operations on it like if we added 2 words-vector (Square(x1+x2)+Square(y1+y2)) we could generate new word related to 2 words and so on .
         ![alt text](image-3.png)
         ![alt text](image-4.png)
         - there are 2 types of word to vec embeddings 
            -CBow and Skipgram(negative-Skip-gram optimized version) 
               - Skip-gram: from one embedding predict N embeddings
               - CBOW : Continues bag of word [ from --N-- embeddings of Words predict one embedding ] 
         - one of problem that we could have  from using one to vec it OOV Problem that th embedding model could work with subsequance of words as new words (The first look)-->couldnt embedding it 
         - So we have other model Called Fast text (we Train model in word as all and its sub words) like Duck--> the training data for model will be Word as all and N-subwords like [Duck,DU,CK,UC,Duc,Uck] Nearly  Could solve OOV(out of Vocapulary problem) solve [each other model is optimized version and so on ]
      # [example of fast text see ](https://www.kaggle.com/code/abdelrahmanm2003/fasttext-embeddings)
#### Now how we encode words for Training Model
   - Simple Machine learning algorithms like ![alt text](image-6.png)
   - Simple DL like Adaline and perceptron [simple shallow nn].
   - complex DL like DeppNN Then we optimized it with new arciticture like some CV ariticture and latest model called sequance models [one example of it Transformer we will see it after some minutes 
   ]
### How NLP Relates to Computer Vision

Let's take an example from computer vision:

* We train a model for each pixel in an image.
* There's a relation between pixels as a whole to specify the image class (e.g., using CNNs with kernels).
* Similarly, we can use CNNs on encoded vectors (embeddings) in NLP. However, for images, the kernel moves in 2 dimensions (2D-CNN).
* Imagine embedded vectors for each word stored in an array. We want to use a kernel to extract data from a number of words (depending on kernel size). Here, we'd use 1D-CNN, but it's not ideal for complex NLP tasks.

Therefore, we need models that can "remember" weights (the knowledge of the model). These are called sequence models, which are the category of most LLMs and NLP models.
   example see it 

 ## [You could see my Notebook Example](https://www.kaggle.com/code/abdelrahmanm2003/sentiment-analysis-ipynb) 
## Sequence Models

* RNN-LSTM-GRU (see [RNN-LSTM-GRU example](https://www.kaggle.com/code/abdelrahmanm2003/gen-rnn-model-ipynb))
* RNN-LSTM-GRU with Attention
* Transformer The Big Boss (The Core Model Of LLM - RAG)-4188-ab77-30a6c2953741.png

### What is Attention?

Attention is a technique that allows the model to understand how important each word is to the current context.


# Transformer

![](https://miro.medium.com/v2/resize:fit:989/0*Q7l2OWtPWiIZzT6T.png)

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
      ![Attention mechanism](<Screenshot From 2025-01-07 19-09-59.png>)
   - Attends to the encoder's output.
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
   ```
   ![alt text](image.png)
3. **Feed-Forward Network**
   - Same as in the encoder.

#  ther is Question may 

#### We take about The architecture But How It actually Trained for Various Tasks (Translation(encoder_decoder_model) , classification(encoder only model),text generation "Trends now adays  😅 "(decoder only models)-->That's type of transformer is the top  used in advanced llm techniques like Rag - Agentic AI by integrate  multible llms together or llms+tools like ote saver ot code runner & evaluator)


# we see in code how transformer built from scratch for text translation

# Now we have the core of LLm but what is LLm Models types (
                                                   - encoder only Models (Bert)
                                                   - decoder only models (like gpt) 
                                                   - Seq2Seq models Encoder-Decoder models

 ) 

 # know How to use Pre trained LLm models .
   - Retrain it from Scratch (use it as arciticture only)
   - Apply Transfer Learning (Fine tuning)
      - [example of fine tuning is Instruction Fine tuning](https://www.kaggle.com/code/abdelrahmanm2003/napoleon-bonapart)
   - or used pretrained mode without training with retriver to retrive data from document using similarity 
      ![alt text](image-7.png) 
   - or intgrate Rag with other llms or tools [Agentic AI](https://github.com/Eng-Abdelrahman-Mostafa-Mohamed/Agents) Inthis project i make Agentic AI for data analysis give it task that you want like visualize som data and it will 
         - generate code 
         - run it 
         - save note and reults and graphs as note 
           