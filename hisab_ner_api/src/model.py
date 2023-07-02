import torch
from transformers import AutoModelForTokenClassification
from transformers import BertForTokenClassification


class HisabNerBertModel(torch.nn.Module):
    """This class build a model model using hagging face
    pretrain model called 'sagorsarker/mbert-bengali-ner' which was
    deployed by sagorsarkar. trained the bert model using 
    Bengali text with 7 different labels.
    """

    def __init__(self):

        super(HisabNerBertModel, self).__init__()
        # self.opt = opt
        self.hisabNerBanglaBert = AutoModelForTokenClassification.from_pretrained(
            "sagorsarker/mbert-bengali-ner", num_labels=7)

    def forward(self, input_id, mask, label):
        output = self.hisabNerBanglaBert(
            input_ids=input_id, attention_mask=mask, labels=label,
            return_dict=False)

        return output
