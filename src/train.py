import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD

from dataloader import DataSequence
from dataloader import load_datasets
from model import HisabNerBertModel, BasicBert
from evaluate import evaluate
from utils import TrainOptions


LEARNING_RATE = 5e-3
EPOCHS = 2
BATCH_SIZE = 2


def train_ner(model, df_train, df_val, opt):

    train_dataset = DataSequence(df_train, opt)
    val_dataset = DataSequence(df_val, opt)

    train_dataloader = DataLoader(
        train_dataset, num_workers=opt.num_workers,
        batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if opt.gpu_ids:
        model = model.cuda()

    for epoch_num in range(opt.n_epochs):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)
        torch.save(model, opt.checkpoints_dir + 'hisab_ner.pth')
    else:
        torch.save(model, opt.checkpoints_dir + 'hisab_ner.pth')
    return model


if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options

    df = load_datasets(opt.dataroot)
    dataset_size = len(df)    # get the number of samples in the dataset.
    print('The number of training samples = %d' % dataset_size)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int(.8 * len(df)), int(.9 * len(df))])
    print(opt.model_name)
    if opt.model_name == 'BanglaBert':
        model = HisabNerBertModel(opt)
    else:
        model = BasicBert(opt)
    model = train_ner(model, df_train, df_val, opt)
    evaluate(model, df_test, opt)
