
import torch
from transformers import BertTokenizerFast, AutoTokenizer

from utils import TrainOptions
from dataloader import ids_exhge_labels


opt = TrainOptions().parse()
if opt.model_name == 'BanglaBert':
    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/mbert-bengali-ner")
else:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
label_all_tokens = False


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


def pred_on_text(model, opt):
    """I-type: transformer model
    r-type: prediction options.
    --------------------------
    return:
           None.
    """
    sentence = []
    # read the test text.
    text = open(opt.pred_text_path, "r", encoding="utf8")
    for line in text:
        sentence.append(line.strip())
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    _, ids_to_labels = ids_exhge_labels(opt)

    text = tokenizer(sentence, padding='max_length',
                     max_length=opt.max_token_length,
                     truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    sentence = sentence[0].split(' ')
    print(prediction_label)
    ner_result = ''
    if len(sentence) == len(prediction_label):
        for idx in range(len(prediction_label)):
            # if prediction_label[idx] == 'U-PERSON':
            #     ner_result += sentence[idx] + ','
            if prediction_label[idx] == 'B-PERSON':
                ner_result += sentence[idx]
            elif prediction_label[idx] == 'I-PERSON':
                ner_result += ' ' + sentence[idx]
        print(ner_result+'\n')
    else:
        print(
            '-------------------------------------------------- \
            ----------------------\n\nInput Text: {}'.format(sentence[0]))
        print('Predicted labels: {}\n'.format(prediction_label))


if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options

    model = torch.load(opt.checkpoints_dir)  # load save model
    # print(model)
    pred_on_text(model, opt)
