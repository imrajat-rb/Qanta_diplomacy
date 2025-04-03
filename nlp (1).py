import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
import jsonlines
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#All the import dependencies ends here

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Removing any gpu wanrings

def diploDTLoad(trainFilePath, testFilePath, valFilePath):
    
    #Loading game dialogue data from .JSONL

    def processFP(fileP):
        data=[]
        with jsonlines.open(fileP) as reader:
            for obj in reader:
                for msg, sender_label in zip(obj['messages'], obj['sender_labels']):
                    data.append({
                        'message': msg,
                        'label': int(sender_label),
                        'speaker': obj['speakers'][0],
                        'receiver': obj['receivers'][0],
                        'game_id': obj['game_id']
                    })
        return(pd.DataFrame(data)) #returning data 

    #calling train, test, validation on processFP()
    trainDTF=processFP(trainFilePath)
    testDTF=processFP(testFilePath)
    valDTF=processFP(valFilePath)

    return(trainDTF, testDTF, valDTF)


def preprocessTextDT(txt): #Cleaning & preprocess text data

    #convert to lowercase
    txt=str(txt).lower()

    #removing specia chars and numbers just for now might help better in undertanding context
    txt=re.sub(r'[^a-zA-Z\s]', '', txt)

    #removing extra whitespaces
    txt=re.sub(r'\s+', ' ', txt).strip()

    return(txt)


def prepingModelDT(dfs, max_words=5000, max_len=150):
    
    comboDF=pd.concat(dfs)

    #preprocess and tokenization
    comboDF['processed_text']=comboDF['message'].apply(preprocessTextDT)
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(comboDF['processed_text'])

    datasets=[]
    for df in dfs:

        df['processed_text']=df['message'].apply(preprocessTextDT)

        #handling sequences
        X=tokenizer.texts_to_sequences(df['processed_text'])
        X=pad_sequences(X, maxlen=max_len)

        #Encoding labels as y
        y=df['label'].astype(int)

        datasets.append((X, y))

    return(datasets, tokenizer)


def modLSTM(max_words, max_len):
    
    model=Sequential([
        
        Embedding(max_words, 64, input_length=max_len), #embedd layer

        #Bi-direction handling context from bith sides
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        Dropout(0.3),

        #dense layers for complex feature extraction
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),

        #Output layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam', loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return(model)


def trainEval(trainX, trainY, testX, testY, valX, valY):
    
    # Model parameters
    max_words=5000
    max_len=150

    model=modLSTM(max_words, max_len)

    #training
    history=model.fit(
        trainX, trainY,
        epochs=5,
        batch_size=32,
        validation_data=(valX, valY),
        verbose=1
    )

    #Evaluate
    y_pred=(model.predict(testX)>0.5).astype(int)

    print("\nTEST set evaluation:")
    print("Accuracy:", accuracy_score(testY, y_pred))
    print("\nClassification Report:")
    print(classification_report(testY, y_pred))

    cm=confusion_matrix(testY, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\nSample predictions are:")
    #choosing 20 random indices here
    sampINDX=np.random.choice(len(testX), 20, replace=False)

    for IDX in sampINDX:
        
        rawTXT=testX[IDX]
        trueLabel=testY[IDX]
        predProb=model.predict(rawTXT.reshape(1, -1))[0][0]
        predLabel = 1 if predProb > 0.5 else 0

        print(f"\nMessage: {rawTXT}")
        print(f"True Label: {trueLabel}")
        print(f"Predicted Label: {predLabel}")
        print(f"Prediction Probability: {predProb:.4f}")
        print("\n")

    return(model, history)


def main():

    #Files Defined here
    trainFilePath='train.jsonl'
    testFilePath='test.jsonl'
    valFilePath='validation.jsonl'

    #Loading dataset
    trainDTF,testDTF,valDTF=diploDTLoad(trainFilePath, testFilePath, valFilePath)

    #Prepping our data
    [(trainX, trainY), (testX, testY), (valX, valY)], tokenizer=prepingModelDT(
        [trainDTF, testDTF, valDTF]
    )

    #Train and Eval
    model, history=trainEval(trainX, trainY, testX, testY, valX, valY)

if __name__ == '__main__':
    main()