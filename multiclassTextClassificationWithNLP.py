from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default
creds, _ = default()
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline


gc = gspread.authorize(creds)

sheet_url = gc.open_by_url('https://docs.google.com/spreadsheets/d/1A8sYQrCHEtUx24VKQIt7pJHfqU12wzV1QA4vyMakILw/edit#gid=416023702')
data = sheet_url.worksheet('bbc-text').get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])

df.info()
df.drop_duplicates()
df.info()
df.sample(5)

def word_tokenize_wrapper(text):
  return word_tokenize(text)

df['tokenize'] = df['text'].apply(word_tokenize_wrapper)

list_stopwords = set(stopwords.words('english'))

def stopwords_removal(words):
  return [word for word in words if word not in list_stopwords]

df['text_stopwords'] = df['tokenize'].apply(stopwords_removal)
df['description'] = df['text_stopwords'].apply(' '.join)
df = df[['text', 'tokenize', 'text_stopwords', 'description', 'category']]
df_fin = df.drop(columns=['text', 'tokenize', 'text_stopwords'])
df_fin.sample(5)

kategori = pd.get_dummies(df_fin.category)
df_new = pd.concat([df_fin, kategori], axis=1)
df_new = df_new.drop(columns='category')
df_new.sample(5)


desc = df_new['description'].values
label = df_new[['business',	'entertainment',	'politics',	'sport',	'tech']].values

train_desc, test_desc, train_label, test_label = train_test_split(desc, label, test_size=0.2)

tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(train_desc)
tokenizer.fit_on_texts(test_desc)

train_sequences = tokenizer.texts_to_sequences(train_desc)
test_sequences = tokenizer.texts_to_sequences(test_desc)

padded_train = pad_sequences(train_sequences)
padded_test = pad_sequences(test_sequences)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.Adam(),
    metrics=['accuracy']
)
model.summary()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>=0.98):
      print('\nEpoch', epoch, '\nGreat!, Accuracy has reached = %2.2f%%' %(logs['accuracy']*100), 'training is already stop!')
      self.model.stop_training = True

num_epochs = 20

history = model.fit(
    padded_train,
    train_label,
    epochs=num_epochs,
    validation_data=(padded_test, test_label),
    batch_size=128,
    verbose=2,
    callbacks = [myCallback()]
)


plt.figure(figsize=(14, 5))
# Accuracy Plot
plt.subplot(1, 2, 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()