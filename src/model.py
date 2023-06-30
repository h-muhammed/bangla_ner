import torch
from transformers import AutoModelForTokenClassification
from transformers import BertForTokenClassification


class HisabNerBertModel(torch.nn.Module):
    """This class build a model model using hagging face
    pretrain model called 'sagorsarker/mbert-bengali-ner' which was
    deployed by sagorsarkar. trained the bert model using 
    Bengali text with 7 different labels.
    """

    def __init__(self, opt):

        super(HisabNerBertModel, self).__init__()
        self.opt = opt
        self.NerBanglaBert = AutoModelForTokenClassification.from_pretrained(
            "sagorsarker/mbert-bengali-ner", num_labels=self.opt.num_labels)

    def forward(self, input_id, mask, label):
        output = self.NerBanglaBert(
            input_ids=input_id, attention_mask=mask, labels=label,
            return_dict=False)

        return output


class BasicBert(torch.nn.Module):
    """This class build a model model using hagging face
    pretrain model called 'bert-base-cased' which was deployed
    by hagging face developers. trained the bert model using 
    English text.
    """

    def __init__(self, opt):
        super(BasicBert, self).__init__()
        self.opt = opt

        self.basicBert = BertForTokenClassification.from_pretrained(
            'bert-base-cased', num_labels=self.opt.num_labels)

    def forward(self, input_id, mask, label):
        output = self.basicBert(
            input_ids=input_id, attention_mask=mask, labels=label,
            return_dict=False)

        return output
