import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
import random
import sys
import time

class Chatbot:

    def __init__(self,json_location):
        with open(json_location) as file:
            self.data = json.load(file)
            self.stemmer = LancasterStemmer()
            self.word_list = []
            self.class_list = []
            self.training_input = np.array([])
            self.training_output = np.array([])
            # print(self.data)

    def create_word_list(self):

        word_list=[]
        class_list=[]
        for data in self.data['response_data']:
            for question in data['question']:
                for word in nltk.word_tokenize(question):
                    word = self.stemmer.stem(word.lower())
                    if word not in word_list and word !="?":
                        word_list.append(word)
            class_list.append(data['class'])
        self.word_list = word_list
        self.class_list = class_list

    def create_bag_of_word(self,sentence):
        sentence_tokens = nltk.word_tokenize(sentence)
        sentence_tokens = [self.stemmer.stem(word.lower()) for word in sentence_tokens if word !="?"]
        bag=[ 1 if (word in sentence_tokens) else 0 for word in self.word_list]
        return bag
    def create_one_hot_encoding(self,class_name):
        one_hot_vector=[]
        for class_name_data in self.class_list:
            if class_name_data == class_name:
                one_hot_vector.append(1)
            else:
                one_hot_vector.append(0)
        return one_hot_vector
    def create_training_data(self):
        # self.create_one_hot_encoding("sad")
        training_input=[]
        training_output=[]
        for data in self.data['response_data']:
            for question in data['question']:
                training_input.append(self.create_bag_of_word(question))
                training_output.append(self.create_one_hot_encoding(data['class']))

        self.training_input = np.array(training_input)
        self.training_output = np.array(training_output)
        # print(self.training_input)
        # print(self.training_output)



    def train(self):
        model = Sequential()
        model.add(Dense(8, activation='sigmoid', input_dim=len(self.training_input[0])))
        model.add(Dense(8, activation='sigmoid'))

        model.add(Dense(len(self.training_output[0]), activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(self.training_input,self.training_output, epochs=4000, batch_size=14, verbose=1, shuffle=True)

        model.save('modelv2.h5')

    def load_chatbot(self):
        model = load_model('modelv2.h5')
        return model


if __name__=='__main__':
    chatbot = Chatbot('./data/response.json')
    chatbot.create_word_list()
    # For training
    # chatbot.create_training_data()
    # chatbot.train()

    # For predicting

    model = chatbot.load_chatbot()
    while True:
        question=input("You: ")
        results = model.predict([chatbot.create_bag_of_word(question)])
        index = np.argmax(results)
        if(np.max(results)<0.5):
            print('Chatbot: Sorry I didn\'t understand it')
            continue
        answer=""
        for data in chatbot.data['response_data']:
            if data['class']==chatbot.class_list[index]:
                answer = random.choice(data['answer'])
        print(f'Chatbot: {answer}')

