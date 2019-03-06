from text_classification_BOW import *
from text_classification_AvgPool import *
from text_classification_GloVe import *
from text_classification_RNN import *
from text_classification_LSTM import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "data"
text_datasets = {x: TextDataset(os.path.join(data_dir, x + ".txt"))
                     for x in ["train", "dev", "test"]}

text_datasets["train"].create_vocab()
text_vocab = text_datasets["train"].vocab

for x in ["train", "dev", "test"]:
        text_datasets[x].encode(text_vocab)

dataloaders = {x: DataLoader(text_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ["train", "dev", "test"]}
dataset_sizes = {x: len(text_datasets[x]) for x in ["train", "dev", "test"]}

unlabelled_dataset = UnlabelledDataset(os.path.join(data_dir, "unlabelled.txt"))
unlabelled_dataset.encode(text_vocab)
unlbl_dataloader = DataLoader(unlabelled_dataset, batch_size=1)


model = train_model(device, dataloaders, dataset_sizes, BOWClassifier(vocab_size=len(text_vocab)))
acc = eval_model(device, dataloaders, dataset_sizes, model)
print('Best test Acc for {:s} model: {:4f}'.format('Bag of Word', acc))
output_file = (data_dir + "/predictions_q1.txt")
predict(device, unlbl_dataloader, model, output_file)


model = train_model(device, dataloaders, dataset_sizes, AvgPoolClassifier(vocab_size=len(text_vocab)))
acc = eval_model(device, dataloaders, dataset_sizes, model)
print('Best test Acc for {:s} model: {:4f}'.format('Avg Pool', acc))
output_file = (data_dir + "/predictions_q2.txt")
predict(device, unlbl_dataloader, model, output_file)


model = train_model(device, dataloaders, dataset_sizes, GloVeClassifier(vocab=(text_vocab)))
acc = eval_model(device, dataloaders, dataset_sizes, model)
print('Best test Acc for {:s} model: {:4f}'.format('GloVe', acc))
output_file = (data_dir + "/predictions_q3.txt")
predict(device, unlbl_dataloader, model, output_file)


model = train_model(device, dataloaders, dataset_sizes, RNNClassifier(vocab=text_vocab))
acc = eval_model(device, dataloaders, dataset_sizes, model)
print('Best test Acc for {:s} model: {:4f}'.format('RNN', acc))
output_file = (data_dir + "/predictions_q4.txt")
predict(device, unlbl_dataloader, model, output_file)


model = train_model(device, dataloaders, dataset_sizes, LSTMClassifier(vocab=text_vocab))
acc = eval_model(device, dataloaders, dataset_sizes, model)
print('Best test Acc for {:s} model: {:4f}'.format('LSTM', acc))
output_file = (data_dir + "/predictions_q5.txt")
predict(device, unlbl_dataloader, model, output_file)