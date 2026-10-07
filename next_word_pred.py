import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("qoute_dataset.csv")
print(df.head())
print(df['quote'][0])
print(df.shape)

quotes = df['quote']
quotes = quotes.str.lower()
import string
translator = str.maketrans('', '', string.punctuation)
quotes = quotes.apply(lambda x : x.translate(translator))
print(quotes.head())

from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore
vocab_size = 8978
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(quotes)
word_index = tokenizer.word_index
print(len(word_index))
print(list(word_index.items())[:10])

sequence = tokenizer.texts_to_sequences(quotes)
print(quotes[0])
print(sequence[0])

X = []
y = []
for seq in sequence:
    for i in range(1, len(seq)):
        input_seq = seq[:i]
        output_seq = seq[i]
        X.append(input_seq)
        y.append(output_seq)

max_len = max(len(x) for x in X)
print(max_len)

from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
x_padded = pad_sequences(X, maxlen = max_len, padding = "pre")
print(x_padded)

y = np.array(y)
from tensorflow.keras.utils import to_categorical #type:ignore
y_one_hot = to_categorical(y, num_classes = vocab_size)

from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN #type:ignore
emb_dim = 50
rnn_units = 128

rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len))
rnn_model.add(SimpleRNN(units = rnn_units))
rnn_model.add(Dense(units=vocab_size, activation='softmax'))
rnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(rnn_model.summary())

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len))
lstm_model.add(LSTM(units = rnn_units))
lstm_model.add(Dense(units=vocab_size, activation='softmax'))
lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(lstm_model.summary())

# history_rnn = rnn_model.fit(x_padded, y_one_hot, epochs = 10, batch_size = 128, validation_split = 0.1)
# history_lstm = lstm_model.fit(x_padded, y_one_hot, epochs = 5, batch_size = 128, validation_split=0.1)
lstm_model.save('lstm_model.h5')

index_to_word = {}
for word, index in word_index.items():
    index_to_word[index] = word

def predictor(model, tokenizer, text, max_len):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen = max_len, padding = 'pre')
    pred = model.predict(seq, verbose = 0)
    pred_index = np.argmax(pred)
    return index_to_word[pred_index]

seed_text = "you will"
next_word = predictor(lstm_model, tokenizer,seed_text, max_len)
print(next_word)

def generate_text(model, tokenizer, seed_text, max_len, n_words):
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, seed_text, max_len)
        if next_word == " ":
            break
        seed_text += " "+ next_word
    return seed_text

seed = "hey i am  "
generate_text = generate_text(lstm_model, tokenizer, seed, max_len, 2)
print(generate_text)
    
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)


