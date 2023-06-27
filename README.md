# Welcome to Hisab Ner


# Datasets pipelining
There are about 3500 annotated text samples. Bellow are the couple of samples.


So, according to the model, we have to deciplane the samples. In this regards, We need to erase the puctuations, as well as 
remove some irregular samples such as bellows:


To some extent, we had to perform some investigate the whole datasets for checking the irregular annotated samples such as bellows:


Eventually, we analyzed and measured the annotated labels quality and nessecities. The original datasets have nearly 20 different 
labels and among them some are unnessesary and we figured out that these are irrelavent to our job. Then, we eradicated them from the labels tags.

For the final datasets, we filtered around 3300 ideal samples. Below are few example of our some ideal samples:




# Installing

For Linux:
Create a virtual environment by below cmd
`python -m venv hisab_ner`
`source hisab_ner/bin/activate`

and then run dependencies in requirement.txt file by below cmd

`pip install -r requirement.txt`


For Windows:



# Project Structure
    
    Hisab_ner
        |___imgs
        |___output
        |___data_preprocess
        |___src
            |___datasets
            |___evaluate.py
            |___model.py
            |___predict.py
            |___requirement.txt
            |___train.py
            |___utils.py
        README.md



# Implementation
Our approach:


    ### training

    ### prediction:

# To-Do
-[] Design and develop an web API for demonstrating the prediction result.
    -[] tools: fastapi
-[] Dockerize the environment for independent platforms.
    -[] tools: docker, docker-compose

# Acknowledgement
Specail thanks goes to Hisab coding test system for assinging and sharing well organized resource and clear instructions. 
[paper](#https://github.com/Rifat1493/Bengali-NER), (#https://arxiv.org/abs/2205.00034)
[Hisab datasets 1](#https://github.com/Rifat1493/Bengali-NER/tree/master/annotated%20data)
[Hisab datasets 2](#https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl)
[Hagging face](#https://huggingface.co/sagorsarker/mbert-bengali-ner)
[medium blogs](#https://medium.com/mysuperai/what-is-named-entity-recognition-ner-and-how-can-i-use-it-2b68cf6f545d)
[misc](#http://nlpprogress.com/english/named_entity_recognition.html), (https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a)
