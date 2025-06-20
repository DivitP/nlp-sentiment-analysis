##Note:set hardware accelerator to gpu
get_ipython().system('pip install transformers')
import os
import gdown
import torch
import numpy as np
import seaborn as sns
import transformers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from sklearn import metrics

from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

get_ipython().run_line_magic('matplotlib', 'inline')


def get_finance_train():
  df_train = pd.read_csv("finance_train.csv")
  return df_train
def get_finance_test():
  df_test = pd.read_csv("finance_test.csv")
  return df_test

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

print ("Train and Test Files Loaded as train.csv and test.csv")

LABEL_MAP = {0 : "negative", 1 : "neutral", 2 : "positive"}
NONE = 4 * [None]
RND_SEED=2020

def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = ["Negative","Neutral","Positive"]
  df_cm = pd.DataFrame(cm,index =labels,columns = labels)
  fig = plt.figure(figsize=(14,12))
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  plt.close()

df_train = get_finance_train() 
df_test = get_finance_test() 

sentences = df_train['Sentence'].values
labels = df_train['Label'].values
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)

tokenizer.vocab_size
print(sentences[0])
tokenized_sentence = tokenizer.tokenize(sentences[0])
print(tokenized_sentence)

print(tokenized_sentence)
tokenizer.convert_tokens_to_ids(tokenized_sentence)

sentences_with_special_tokens = []
for i in sentences:
  new_str = "[CLS] " + i + " [SEP]"
  sentences_with_special_tokens.append(new_str)

tokenized_texts = []
for i in sentences_with_special_tokens:
  token_str = tokenizer.tokenize(i)
  tokenized_texts.append(token_str)

input_ids = []
for i in tokenized_texts:
  id_str = tokenizer.convert_tokens_to_ids(i)
  input_ids.append(id_str)

input_ids = pad_sequences(input_ids,
                          maxlen=128, 
                          dtype="long",
                          truncating="post",
                          padding="post")
print(input_ids[0])

attention_masks = []
for i in input_ids:
  mask=[float(j>0) for j in i]
  attention_masks.append(mask)

X_train, X_val, y_train, y_val = train_test_split(input_ids, labels, test_size=0.15, random_state=RND_SEED) 
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=0.15, random_state=RND_SEED)


#Convert data to tensors and create DataLoaders
train_inputs = torch.tensor(np.array(X_train))
validation_inputs = torch.tensor(np.array(X_val))
train_masks = torch.tensor(np.array(train_masks))
validation_masks = torch.tensor(np.array(validation_masks))
train_labels = torch.tensor(np.array(y_train))
validation_labels = torch.tensor(np.array(y_val))

batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data); 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data); 
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#BERT model with a single linear classification layer on top; bert-base-uncased = 12-layer BERT small model w/ uncased vocab
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 3,
    output_attentions = False, # dont return attentions weights.
    output_hidden_states = False, # dont return hidden-states.
)

model.cuda() # run via gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)


optimizer = AdamW(model.parameters(),
                  lr = 5e-5, ### prev was 2e-5
                  eps = 1e-8
                )
epochs = 4


total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

training_loss = []
validation_loss = []
training_stats = []
for epoch_i in range(0, epochs):
    print('Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training the model')
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 20 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        # Unpack this training batch and copy each tensor to the GPU 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        #  clear prev calculated gradients before doing the backward pass 
        model.zero_grad()

        # evaluate model on the training batch).
        outputs = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        total_train_loss += loss.item()

        # calc gradients through backward pass 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("Evaluating on Validation Set")
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
    
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("Validation Loss: {0:.2f}".format(avg_val_loss))

    training_loss.append(avg_train_loss)
    validation_loss.append(avg_val_loss)

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy

        }
    )

print("Training complete!")



fig = plt.figure(figsize=(12,6))
plt.title('Loss over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(training_loss, label="train")
plt.plot(validation_loss, label="validation")
plt.legend()
plt.show()


test_sentences= df_test['Sentence'].values
test_labels = df_test['Label'].values
test_input_ids, test_attention_masks = [], []

test_sentences = ["[CLS] " + sentence + " [SEP]" for sentence in test_sentences]
tokenized_test_sentences = [tokenizer.tokenize(sent) for sent in test_sentences]
test_input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_test_sentences]

test_input_ids = pad_sequences(test_input_ids,
                               maxlen=128,
                               dtype="long",
                               truncating="post",
                               padding="post")

for sequence in test_input_ids:
  mask = [float(i>0) for i in sequence]
  test_attention_masks.append(mask)


batch_size = 32
test_input_ids = torch.tensor(test_input_ids)
test_attention_masks = torch.tensor(test_attention_masks)
test_labels = torch.tensor(test_labels)
prediction_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))
model.eval()

predictions , true_labels = [], []

for batch in prediction_dataloader:
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_labels = batch

  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  predictions.append(logits)
  true_labels.append(label_ids)

y_logits, y_true, y_preds = [], [], []

for chunk in predictions:
  for logits in chunk:
    y_logits.append(logits)

for chunk in true_labels:
  for label in chunk:
    y_true.append(label)

for logits in y_logits:
  y_preds.append(np.argmax(logits))

print ('Test Accuracy: {:.2%}'.format(metrics.accuracy_score(y_preds,y_true)))
plot_confusion_matrix(y_true,y_preds)