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
There are about 3500 annotated text samples. Below are a couple of samples: <br/>
```
["অগ্রণী ব্যাংকের জ্যেষ্ঠ কর্মকর্তা পদে নিয়োগ পরীক্ষার প্রশ্নপত্র ফাঁসের অভিযোগ উঠেছে।", ["B-ORG", "L-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]]
["ব্যাংকের চেয়ারম্যানও এ অভিযোগের সত্যতা স্বীকার করেছেন।", ["O", "O", "O", "O", "O", "O", "O", "O"]]
```
<br/>
However, according to our chosen model, we have to discipline teh original text. To some extent, we were performed a thorough  investigate among the datasets for checking the irregular annotated samples
<br/>
Eventually, we analyzed and measured the annotated labels quality and nessecities. The original datasets have nearly 20 different 
labels and among them some are unnessesary and we figured out that these are irrelavent to our job. Thus, we eradicated them from the samples annotations. We chosed 7 different labels for our model.
<br/>
For the final datasets, we filtered around 3467 ideal samples. Below are fw example of our some ideal samples:  <br/>
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
#### Initial Thought
Since our job was to identify the persons name from the given text, so it can be easily done by text classification task. Whereas, datasets has different labels, perhaps, 20 different labels at most, we decided to design a model for multi-label classification task.  <br/>
#### Model Design
There are neumerous number of methodologies available out there to solve such sort of problems like as `basic probabilistic approach, rnn, lstm, transformer etc`. Amongst them transformer is the sota model such kind of multi-label jobs. So, we decided to implement a `transfer learning` method for viable solutions. We have a couple of options for choosing pretrain transformer such as `bert-base-cased` and `sagorsarker/mbert-bengali-ner` from hagging face. We implemented both model and `sagorsarker/mbert-bengali` bert model generated remarkable performance as we showed in loss graph. Later model perform well due to trained by large amount of Bengali text corpus for the specific job by `sagorsarkar`. The pretrained model deployed in hagging face hub. <br/>
Below are the model codesnipat backed by `sagorsarker/mbert-bengali-ner` hagging face pretrain model. <br/>
```python
class HisabNerBertModel(torch.nn.Module):
    """This class build a model model using hagging face
    pretrain model called 'sagorsarker/mbert-bengali-ner' which was
    deployed by sagorsarkar. trained the bert model using 
    Bengali text with 7 different labels.
    """
    def __init__(self, opt):
       
        super(HisabNerBertModel, self).__init__()
        self.opt = opt
        self.hisabNerBanglaBert = AutoModelForTokenClassification.from_pretrained("sagorsarker/mbert-bengali-ner", num_labels=self.opt.num_labels)
       
    def forward(self, input_id, mask, label):
        output = self.hisabNerBanglaBert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        
        return output
```

#### Optimization
Since our job was to detect the name from text, we noticed that the train datasets has lots of labels and most of them are not required for our task. Hence, we decided to shrink the labels to minimized the number labels which was 7 at the end. The final annotated labels are `['B-PERSON', 'GPE', 'I-PERSON', 'LAW', 'O', 'ORG', 'U-PERSON']`. At the begining the number of labels are around 20.




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
