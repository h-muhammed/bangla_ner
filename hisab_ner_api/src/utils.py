import logging
import logging.config
import torch
from transformers import AutoTokenizer
from functools import lru_cache


tokenizer = AutoTokenizer.from_pretrained("sagorsarker/mbert-bengali-ner")
ids_to_labels = {0: 'GPE', 1: 'O', 2: 'LAW', 3: 'I-PERSON', 4: 'ORG',
                 5: 'B-PERSON', 6: 'U-PERSON'}
label_all_tokens = False


@lru_cache(maxsize=256)
def align_word_ids(texts):
    """i-type: text
    r-type: list[int]
    """
    tokenized_inputs = tokenizer(
        texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except Exception as ex:
                label_ids.append(-100)
                # print(ex)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except Exception as ex:
                label_ids.append(-100)
                # print(ex)
        previous_word_idx = word_idx

    return label_ids


def hisab_ner(model, sentence):
    """I-type: transformer model
    r-type: prediction options.
    --------------------------
    return:
           None.
    """
    info = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    text = tokenizer(sentence, padding='max_length',
                     max_length=512,
                     truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    token = sentence
    tokens = token.split(' ')
    # print(prediction_label)
    ner_result = ''
    if len(tokens) == len(prediction_label):
        for idx in range(len(prediction_label)):
            # if prediction_label[idx] == 'U-PERSON':
            #     ner_result += sentence[idx] + ','
            if prediction_label[idx] == 'B-PERSON':
                ner_result += tokens[idx]
            elif prediction_label[idx] == 'I-PERSON':
                ner_result += ' ' + tokens[idx]
        print(ner_result+'\n')
    else:
        print(
            '-------------------------------------------------- \
            ----------------------\n\nInput Text: {}'.format(sentence))
        print('Predicted labels: {}\n'.format(prediction_label))
    info.append({
        'text': sentence,
        'name_entiry': ner_result
    })
    return info
