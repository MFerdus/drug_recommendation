# Drug Recommendation System (Deep Learning + NLP)
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

#Data Reading
df = pd.read_csv('drug_reviews.csv')
df.head()


# Preprocess text and labels

def normalize_text(text str) - str
    if text is None
        return ''
    text = str(text).lower()
    text = re.sub(rhttpS+wwwS+,  , text)
    text = re.sub(r[^a-z0-9s]+,  , text)
    text = re.sub(rs+,  , text).strip()
    return text

X_text = (df['condition'].fillna('') + ' ' + df['review'].fillna('')).astype(str).apply(normalize_text)
y = df['drugName'].astype(str).str.strip()

mask = (X_text.str.len()  0) & (y.str.len()  0)
X_text, y = X_text[mask], y[mask]

# filter rare classes for stability
MIN_SAMPLES = 30
vc = y.value_counts()
keep = vc[vc = MIN_SAMPLES].index
X_text, y = X_text[y.isin(keep)], y[y.isin(keep)]

le = LabelEncoder()
y_enc = le.fit_transform(y)

print('Samples', len(y_enc))
print('Classes', len(le.classes_))


# Traintest split
X_train, X_test, y_train, y_test = train_test_split(
    X_text.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Build model (TextVectorization inside model)
MAX_TOKENS = 40000
SEQ_LEN = 200
EMBED_DIM = 128
LSTM_UNITS = 64

text_in = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=SEQ_LEN,
    standardize=None
)

x = vectorizer(text_in)
x = tf.keras.layers.Embedding(MAX_TOKENS, EMBED_DIM)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(len(le.classes_), activation='softmax')(x)

model = tf.keras.Model(text_in, out)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Training
BATCH_SIZE = 64
EPOCHS = 5

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
eval_loss, eval_acc = model.evaluate(test_ds)
print('Test accuracy', round(float(eval_acc), 4))

# Top-K recommendations
def top_k(condition, review, k=5)
    text = normalize_text(condition + ' ' + review)
    probs = model.predict(np.array([[text]]), verbose=0)[0]
    idx = np.argsort(probs)[-1][k]
    return [(le.classes_[i], float(probs[i])) for i in idx]

top_k('GERD', 'Heartburn after meals, worse at night', k=5)

# Save model + classes