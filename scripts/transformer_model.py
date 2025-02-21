import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, MultiHeadAttention, Embedding, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import os
import tarfile
import glob

def download_and_extract_data(url, filename, extract_to):
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(response.content)
    if not os.path.exists(extract_to):
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=extract_to)

def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in glob.glob(os.path.join(dir_name, '*.txt')):
            with open(fname, encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(1 if label_type == 'pos' else 0)
    return texts, labels

def preprocess_data(texts, tokenizer, maxlen):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

def predict_text(model, tokenizer, text, maxlen):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    prediction = model(tf.constant(padded_sequence))
    label = decode_prediction(prediction)
    return label

def decode_prediction(prediction):
    label = tf.argmax(prediction, axis=1).numpy()[0]
    return 'Positive' if label == 1 else 'Negative'

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerClassifier(Model):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = Dropout(0.1)
        self.dense = Dense(20, activation="relu")
        self.out = Dense(2, activation="softmax")

    def call(self, inputs, training=None):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x, training=training)
        x = self.global_avg_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return self.out(x)

# Parameters
maxlen = 100
vocab_size = 20000
embed_dim = 32
num_heads = 2
ff_dim = 32

# Download and preprocess data
imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
data_dir = "aclImdb"
download_and_extract_data(imdb_url, "aclImdb_v1.tar.gz", data_dir)

# Load raw data
train_texts, train_labels = load_imdb_data(os.path.join(data_dir, "aclImdb/train"))

# Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

# Preprocess data
x_train = preprocess_data(train_texts, tokenizer, maxlen)
y_train = tf.convert_to_tensor(train_labels)

# Model
model = TransformerClassifier(maxlen, vocab_size, embed_dim, num_heads, ff_dim)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

# Example prediction
sample_pos_text = "The movie was fantastic, I really enjoyed it!"
prediction_label = predict_text(model, tokenizer, sample_pos_text, maxlen)
print(f"Prediction for pos text: {prediction_label} \n")

sample_neg_text = "it's the worst movie ever!!"
prediction_label = predict_text(model, tokenizer, sample_neg_text, maxlen)
print(f"Prediction for neg text: {prediction_label} \n")
