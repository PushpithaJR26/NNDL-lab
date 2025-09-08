from tensorflow.keras.datasets import imdb
 from tensorflow.keras.preprocessing.sequence import pad_sequences
 # Load dataset
 (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
 # Pad sequences to ensure equal length inputs
 X_train = pad_sequences(X_train, maxlen=200)
 X_test = pad_sequences(X_test, maxlen=200)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128, 
input_length=200),
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dense(1, activation='sigmoid')
 ])
 model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])
 history = model.fit(X_train, y_train, epochs=10, batch_size=64, 
validation_data=(X_test, y_test))
 import matplotlib.pyplot as plt
 # Plot accuracy
 plt.plot(history.history['accuracy'], label='Training Accuracy')
 plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
 plt.xlabel('Epochs')
 plt.ylabel('Accuracy')
 plt.legend()
