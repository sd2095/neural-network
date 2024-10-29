import re
import os
import json
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import keras
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')  # defining stop_words
stop_words.remove('not')
lemmatizer = WordNetLemmatizer()


def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words_to_remove])

def cleanData(list_of_statements):
    stop_words = set(stopwords.words('english'))
    essential_stopwords = {'not', 'no', 'very', 'only', 'but'}
    stop_words_to_remove = stop_words - essential_stopwords

def nlpPreprocessing(tweet_list):
    # Convert the list to a string
    tweet = ' '.join(tweet_list)
    tweet = re.sub(r'[^\w\s]', '', tweet)  #remove punctuations and characters

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenization
    tokens = nltk.word_tokenize(tweet)  # Convert text to tokens

    # Remove single-character tokens (except meaningful ones like 'i' and 'a')
    tokens = [word for word in tokens if len(word) > 1]

    # Remove stopwords
    tweet = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tweet = [lemmatizer.lemmatize(word) for word in tweet]

    # Join words back into a single string
    tweet = ' '.join(tweet)
    return tweet


def read_json_from_folder(folder_path):
    json_data_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):  # Only process JSON files
            file_path = os.path.join(folder_path, file_name)

            # Open and read each JSON file
            with open(file_path, 'r', encoding='utf-8') as json_file:
                try:
                    print(f"Loading file {file_path}")
                    data = json.load(json_file)  # Load the JSON data

                    # Append the JSON data to the list
                    json_data_list.append(data)

                except json.JSONDecodeError:
                    print(f"Error reading {file_name}")

    return json_data_list


def printClassificationReport(model, X_train, X_test, y_train, y_test):
    predictions = model.predict(X_train)
    print("Train Data Classification report")
    print(classification_report(y_train, predictions))

    # Test Data Results
    predictions = model.predict(X_test)
    print("Test Data Classification report")
    print(classification_report(y_test, predictions))


def runSVCModel(X_train, X_test, y_train, y_test):
    model = LinearSVC()
    model.fit(X_train, y_train)
    printClassificationReport(model, X_train, X_test, y_train, y_test)


def runRandomForestClassifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    printClassificationReport(model, X_train, X_test, y_train, y_test)


def runDecisionTreeClassifier(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    printClassificationReport(model, X_train, X_test, y_train, y_test)


def runMultiLayerPerceptronWithBCE(X_train, X_test, y_train, y_test):
    model = keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history

def runCNN(X_train, X_test, y_train, y_test):
    model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[1], 1)),  # Convolutional layer
        MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
        Conv2D(X_train.shape[1], (3, 3), activation='relu'),  # Second convolutional layer
        MaxPooling2D(pool_size=(2, 2)),  # Second max pooling layer
        Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
        Flatten(),  # Flatten the output
        Dense(64, activation='relu'),  # Fully connected layer
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history

def runSVCMultiLabel(X_train, X_test, y_train, y_test):
    svc = SVC(kernel='linear', probability=True)
    # Use One-vs-Rest strategy
    model = OneVsRestClassifier(svc)
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred_test = model.predict(X_test)
    # Make predictions on the training set
    y_pred_train = model.predict(X_train)
    # Calculate accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train, average='macro')
    f1_test = f1_score(y_test, y_pred_test, average='macro')

    # Print accuracy and F1 scores
    print(f"Training Accuracy: {accuracy_train:.2f}")
    print(f"Test Accuracy: {accuracy_test:.2f}")
    print(f"Training F1 Score (macro): {f1_train:.2f}")
    print(f"Test F1 Score (macro): {f1_test:.2f}")

def runMultiLabelClassification(X_train, X_test, y_train, y_test):
    # Step 5: Build the neural network model
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),  # Prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')  # 22 output nodes with sigmoid for multi-label
    ])

    # Compile the model with binary cross-entropy loss for multi-label classification
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history


def runMultiLabelClassificationWithBatchNormalization(X_train, X_test, y_train, y_test):
    # Step 5: Build the neural network model
    model = Sequential([
        Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='relu')  # Output layer remains the same
    ])

    # Compile the model with binary cross-entropy loss for multi-label classification
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history

def runMultiClassification(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.0002)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history

def runLSTM(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[0], X_train.shape[1]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history

def plotTrainAccuracyAndLoss(modelhistory):
    # Visualize the training and validation metrics
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(modelhistory.history['accuracy'], label='Train Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(modelhistory.history['loss'], label='Train Loss')
    plt.plot(modelhistory.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def runLSTMModel(X_train):
    # Step 5: Build the neural network model
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),  # Prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(23, activation='relu')  # 23 output nodes with sigmoid for multi-label
    ])

    # Compile the model with binary cross-entropy loss for multi-label classification
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=[AUC(name="auc")])
