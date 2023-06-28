# Welcome to Hisab Ner
![alt-text-1](image1.png "title-1") ![alt-text-2](image2.png "title-2")


# Installing

For Linux: <br />
Create a virtual environment by below cmd <br />
```
python -m venv hisab_ner
```
For activation <br />
```
source hisab_ner/bin/activate
``` 
And then install relevant dependencies by below cmd <br />

```
pip install -r requirement.txt
```


For Windows:


# Datasets pipelining
There are about 3500 annotated text samples. Bellow are the couple of samples.
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)


So, according to the model, we have to disciplineregardpunctuationsBelownecessitiesunnecessaryirrelevantlabelexamples the samples. In this regards, We need to erase the puctuations, as well as 
remove some irregular samples such as bellows:


To some extent, we had to perform some investigate the whole datasets for checking the irregular annotated samples such as bellows:


Eventually, we analyzed and measured the annotated labels quality and nessecities. The original datasets have nearly 20 different 
labels and among them some are unnessesary and we figured out that these are irrelavent to our job. Then, we eradicated them from the labels tags.

For the final datasets, we filtered around 3300 ideal samples. Below are fw example of our some ideal samples:
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)



# Project Structure
    
    Hisab_ner
        |___imgs
        |___output
        |___data_preprocess
        |         |___text_process.py
        |         |___Hisab_Ner.txt
        |___src
            |___datasets
            |      |____hisab_ner.csv
            |      |____pred_text.txt
            |___evaluate.py
            |___model.py
            |___predict.py
            |___requirement.txt
            |___train.py
            |___utils.py
        README.md



# Implementation
### Our approach:


#### Train <br />
For gpu, add `--gpu_ids 1,2, or 3` etc. For cpu, `-1` <br/>
```
python train.py --dataroot datasets/ner.csv --model_name BanglaBert --gpu_ids -1
```
<br/>

#### Inference  <br/> 
Put inference text in `datasets/pred_text.txt`  <br/>  
```
python predict.py --modle_name BanglaBert --gpu_ids -1
```

# Whats Next i.e: To-Do  <br/>

- [ ] Design and develop an web API for demonstrating the prediction result. <br/>
    - [ ] tools: fastapi <br/>
- [ ] Dockerize the environment for independent platforms.  <br/>
    - [ ] tools: docker, docker-compose <br/>

# Acknowledgement
Specail thanks goes to Hisab coding test system for assinging and sharing a well organized resource and clear instructions. <br/> <br/>
[paper](https://arxiv.org/abs/2205.00034)  <br/>
[Bengali_Ner](https://github.com/Rifat1493/Bengali-NER)  <br/>
[Hisab datasets 1](https://github.com/Rifat1493/Bengali-NER/tree/master/annotated%20data)  <br/>
[Hisab datasets 2](https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl)  <br/>
[Hagging face](https://huggingface.co/sagorsarker/mbert-bengali-ner)  <br/>
[medium blogs](https://medium.com/mysuperai/what-is-named-entity-recognition-ner-and-how-can-i-use-it-2b68cf6f545d)  <br/>
[misc](http://nlpprogress.com/english/named_entity_recognition.html) <br/>
https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
