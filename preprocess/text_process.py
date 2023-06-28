import pandas as pd
import numpy as np


def preprocess(file_path):
  """
  i-type: txt file path
  r-type: text sentences and corresponding labels
  """
  lines = []
  text = open(file_path, "r", encoding="utf8")
  for line in text:
       lines.append(line.replace('[', '').replace(']', '').replace('"', '').replace('"', '').replace('(','').replace('...', '').replace('”', ''). \
                 replace("'",'').replace('U-DATE','O').replace('L-PERSON', 'I-PERSON').replace('U-GPE','GPE').replace('B-GPE','GPE').replace('I-GPE','GPE'). \
                 replace('L-GPE','GPE').replace('B-LAW','LAW').replace('I-LAW','LAW').replace('L-LAW','LAW').replace('B-ORG','ORG').replace('I-ORG','ORG'). \
                 replace('L-ORG','ORG').replace('U-ORG','ORG').replace('\n','').strip())

  text_corpus = []
  for line in lines:
    text_corpus.append(line.split('।'))

  corpus = []
  for idx in range(len(text_corpus)):
    if len(text_corpus[idx]) == 2:
      corpus.append(text_corpus[idx])
  
  sentences, labels = [], []
  for idx in range(len(corpus)):
    if len(corpus[idx]) == 2:
      sentences.append(corpus[idx][0])
      labels.append(corpus[idx][1])
  
  new_labels = []
  for idx in range(len(labels)):
    temp_label = labels[idx]
    temp_label = temp_label.replace(',','')
    new_labels.append(''.join(str(x) for x in temp_label).strip())
  
  return sentences, new_labels

if __name__ == '__main__':

    txt_path = "Hisab_Ner.txt"

    sentences, labels = preprocess(txt_path) 
    df = pd.DataFrame(list(zip(sentences, labels)),
                columns =['text', 'labels'])

    #save the dataframe to csv file
    csv_path = '..\src\datasets\preprocessed_hisab_ner_text_test.csv'
    df.to_csv(csv_path, index=False)
