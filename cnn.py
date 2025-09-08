 import numpy as np
 import tensorflow as tf
 from tensorflow.keras.datasets import cifar10
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
 from sklearn.metrics import accuracy_score, precision_score, recall_score
 # Load CIFAR-10 dataset
 (X_train, y_train), (X_test, y_test) = cifar10.load_data()
 # Normalize pixel values to be between 0 and 1
 X_train = X_train.astype('float32') / 255.0
 X_test = X_test.astype('float32') / 255.0
 # Convert class vectors to binary class matrices (one-hot encoding)
 y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
 y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
 # Define the CNN model
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
input_shape=(32, 32, 3)))
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
 model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dense(10, activation='softmax'))
 # Compile the model
 model.compile(optimizer='adam', 
loss='categorical_crossentropy',metrics=['accuracy'])
# Train the model
 model.fit(X_train, y_train, epochs=10, batch_size=64, 
validation_data=(X_test, y_test))
 # Evaluate the model
 loss, accuracy = model.evaluate(X_test, y_test)
 print('Test accuracy:', accuracy)
 # Make predictions on the test set
 y_pred_prob = model.predict(X_test)
 y_pred = np.argmax(y_pred_prob, axis=1)
 y_true = np.argmax(y_test, axis=1)
 # Calculate precision and recall
 precision = precision_score(y_true, y_pred, average='macro')
 recall = recall_score(y_true, y_pred, average='macro')
 print('Precision:', precision)
 print('Recall:', recall)
