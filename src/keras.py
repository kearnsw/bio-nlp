import numpy as np
from src.preprocess import load_data
from src.utils import idx_tags, seq_to_one_hot, idx_words, text2seq, tags2idx
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional

max_len = 250
docs, labels = load_data("../data/cadec/text/", "../data/cadec/original/")
print(len(docs))
x_train = docs[:1000]
y_train = labels[:1000]

x_test = docs[1001:1200]
y_test = labels[1001:1200]

word2idx = idx_words(x_train)
print(word2idx)
tag2idx = idx_tags(y_train)

x_train = sequence.pad_sequences([text2seq(doc, word2idx) for doc in x_train], maxlen=max_len)
x_test = sequence.pad_sequences([text2seq(doc, word2idx) for doc in x_test], maxlen=max_len)
y_train = np.array([seq_to_one_hot(tags2idx(seq, tag2idx), (len(seq), len(tag2idx))) for seq in y_train])
y_test = np.array([seq_to_one_hot(tags2idx(seq, tag2idx), (len(seq), len(tag2idx))) for seq in y_test])

print(y_test[0].shape)
num_classes = len(tag2idx)
model = Sequential()
model.add(Embedding(20000, 128, input_length=max_len))
model.add(LSTM(64, return_sequences=True))
model.add(Dense(num_classes, activation='softmax'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=10,
          batch_size=1000,
          validation_data=[x_test, y_test])
