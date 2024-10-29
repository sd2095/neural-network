import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import read_json_from_folder, nlpPreprocessing, stop_words, runSVCModel, runRandomForestClassifier, \
    runDecisionTreeClassifier, plotTrainAccuracyAndLoss, runCNN, runSVCMultiLabel, runMultiLabelClassification, \
    runMultiLabelClassificationWithBatchNormalization, runLSTM, runMultiClassification, runMultiLayerPerceptronWithBCE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

folder_path = 'ECHR_Dataset/EN_train'
folder_path2 = 'ECHR_Dataset/EN_test'

# Read JSON files from both folders
print("Started reading json files for train data")
train_data = read_json_from_folder(folder_path)
print("Started reading json files for test data")
test_data = read_json_from_folder(folder_path2)

# Combine the data from both folders into one list
combined_data = train_data + test_data
df = pd.DataFrame(combined_data)
print("Completed framing data into df")

combined_text = " ".join([" ".join(row) for row in df['TEXT']])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

# Display the generated word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis
plt.show()


##Create Violation/Non Violation target column
# df['Violation_Status'] = df['VIOLATED_ARTICLES'].apply(lambda x: 1 if len(x) > 0 else 0)
# df['TEXT'] = df['TEXT'].apply(nlpPreprocessing)
# df['TEXT_LENGTH'] = df['TEXT'].apply(lambda x: len(x))
# df = df[df['TEXT_LENGTH'] <= 50000]


vect = TfidfVectorizer(max_features=5000)
X = vect.fit_transform(df.TEXT).toarray()


y = df.Violation_Status
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

## Violation Status binary class classification
# runSVCModel(X_train, X_test, y_train, y_test)
# runRandomForestClassifier(X_train, X_test, y_train, y_test)
# runDecisionTreeClassifier(X_train, X_test, y_train, y_test)
# modelHistory = runMultiLayerPerceptronWithBCE(X_train, X_test, y_train, y_test)
# modelHistory = runCNN(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistory)
# runSVCMultiLabel(X_train, X_test, y_train, y_test)


# mlb = MultiLabelBinarizer()
# X = vect.fit_transform(df.TEXT).toarray()
# y = df.VIOLATED_ARTICLES
# y_filled = [item if isinstance(item, list) else [] for item in y]
# y_filled = mlb.fit_transform(y_filled)
# X_train, X_test, y_train, y_test = train_test_split(X, y_filled, test_size=0.2, random_state=42)
#
# modelHistoryMLC = runMultiLabelClassification(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistoryMLC)

#modelHistoryMLB = runMultiLabelClassification(X_train_mlb, X_test_mlb, y_train_mlb, y_test_mlb)
#plotTrainAccuracyAndLoss(modelHistoryMLB)

#modelHistoryMLC = runMultiLabelClassificationWithBatchNormalization(X_train_mlb, X_test_mlb, y_train_mlb, y_test_mlb)
#plotTrainAccuracyAndLoss(modelHistoryMLC)



# case importance - multi class classification
# one_hot_encoded = pd.get_dummies(df['IMPORTANCE'], prefix='importance')
# one_hot_encoded = one_hot_encoded.astype(int)
#
# X = vect.fit_transform(df.TEXT)
# y = one_hot_encoded.values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# modelHistoryMLC = runMultiClassification(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistoryMLC)
