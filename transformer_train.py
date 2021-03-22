# Imports 
import librosa
import librosa.display
import librosa.effects
import librosa.util

import numpy as np
import sys, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformer_models import *
from transformer_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_features', dest='num_features', type=int, default=8, help="Number of features to use (6 or 8)")
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Number of epochs to use')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help="batch size")
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--num_layers', dest='num_layers', type=int, default=2, help='Number of MLP layers to stack')
parser.add_argument('--run', dest='run', type=int, default=0, help='Experiment run number')
parser.add_argument('--model', dest='model', default='EmotionTransformerPrototype', type=str, help='Model to use to run experiment')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2, help='dropout probability')
args = parser.parse_args()

path = '/home/CREMA-D/AudioWAV/'
files = os.listdir(path)

# Helper function from homework
def pad_wav(wav, wav_max_length, pad=0):
    """Pads audio wave sequence to be `wav_max_length` long."""
    dim = wav.shape[1]
    padded = np.zeros((wav_max_length, dim)) + pad
    if len(wav) > wav_max_length:
        wav = wav[:wav_max_length]
    length = len(wav)
    padded[:length, :] = wav
    return padded, length

summary = pd.read_csv('/home/CREMA-D/processedResults/summaryTable.csv')

num_files = len(os.listdir(path)) #not sure how you want to count files
count = 0

# Aim to get to 12 features
num_features = args.num_features

# Keep track of min and max duration of all data
min_dur = np.inf
max_dur = 0
max_length = 0

print(f'Loading the data...')
# mel spectrogram is padded to 250 (cause max duration is ~5 seconds, which is near 250), and n_mels is default 128
X = np.zeros((num_files, num_features + 250 * 128))
Y = np.zeros(num_files).astype(str)
for sample in tqdm(files): #depends on how you access
    file = os.path.join(path,sample)
    current_wav, current_sr = librosa.load(file) #fix for set up
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=current_wav, sr=current_sr)
    m_log = librosa.power_to_db(mel_spec)
    m_log_norm = librosa.util.normalize(m_log)
    padded_wav, input_length = pad_wav(m_log_norm.T, 250)
    flat_mel = padded_wav.flatten()
    
    # Prosodic features
    f0_series = librosa.yin(current_wav, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'))
    rms_series = librosa.feature.rms(y=current_wav)
    f0_max = np.amax(f0_series)
    f0_min = np.amin(f0_series)
    # Get f0 range
    f0_range = f0_max - f0_min
    # duration
    duration = librosa.get_duration(y=current_wav, sr=current_sr)

    # Outer duration
    if duration > max_dur:
        max_dur = duration
    if duration < min_dur:
        min_dur = duration

    f0_mean = np.mean(f0_series)
    rms_max = np.amax(rms_series)
    rms_min = np.amin(rms_series)
    rms_mean = np.mean(rms_series)

    if num_features == 8:
        x = np.array([f0_min, f0_max, f0_mean, f0_range, duration, rms_min, rms_max, rms_mean])
    else:
        x = np.array([f0_min, f0_max, f0_mean, rms_min, rms_max, rms_mean])
    x = np.append(x, flat_mel)
    
    X[count,:] = x
    # Get the label for VoiceVote
    info = summary.loc[summary['FileName'] == sample.split('.')[0]]
    try:
        Y[count] = info['VoiceVote'].values[0]
    except Exception as ex:
        print(f'info: {info}')
        print(f'index count: {count}')
        index = count
        print(f'unable to find file: {sample}')
        count -= 1
    count += 1
print(f'shape of train data: {X.shape}')
print(f'shape of labels: {Y.shape}')

# Remove that one example without a label
X = np.delete(X,-1,axis=0)
Y = Y[:-1]
print(f'New X shape: {X.shape}')
print(f'New Y shape: {Y.shape}')

# Find number of unique labels
num_unique = np.unique(Y).shape[0]
print(f'num classes: {num_unique}')

# Use label encoder for string labels
le = preprocessing.LabelEncoder()
le.fit(Y)
print(f'classes: {le.classes_}')
transformed_labels = le.transform(Y)
print(f'shape of transformed labels: {transformed_labels.shape}')

# Set up the dataloaders
train_dataset = CREMADataset(X, transformed_labels, X.shape[0], split='train')
val_dataset = CREMADataset(X, transformed_labels, X.shape[0], split='val')
test_dataset = CREMADataset(X, transformed_labels, X.shape[0], split='test')

print('Train dataset: ', len(train_dataset))
print('Val dataest: ', len(val_dataset))
print('Test dataset: ', len(test_dataset))

# Set dataloaders
batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = args.model
lr = args.lr
weight_decay = args.weight_decay
# Set up the model
if model == 'EmotionTransformerPrototype':
    model = EmotionTransformerPrototype(num_features + 250 * 128, num_unique, num_layers=args.num_layers).cuda()
elif model == 'EmotionTransformerPrototypeImproved':
    model = EmotionTransformerPrototypeImproved(num_features + 250 * 128, num_unique, num_layers=args.num_layers).cuda()
elif model == 'EmotionTransformerPrototypeMLP':
    model = EmotionTransformerPrototypeMLP(num_features + 250 * 128, num_unique, num_layers=args.num_layers, dropout=args.dropout).cuda()

# Set to negative log likelihood loss and Adam optimizer
criterion = nn.NLLLoss()
optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# start training
losses = []
val_loss = []
accuracies = []
val_accuracy = []

epochs = args.epochs
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0
    correct_count = 0
    for x,y in train_loader:
        optim.zero_grad()

        x_cuda = x.cuda()
        y_cuda = torch.squeeze(y).cuda()

        # Output from mode
        output = model(x_cuda)
        output = output.cuda()

        # Calculate loss
        loss = criterion(output, y_cuda)

        # Calculate predictions
        # Need to take max over the log probs (batch_size, num_classes)
        _, pred = torch.max(output, 1)
        pred = pred.type(torch.FloatTensor).cuda()

        num_correct = np.sum(y_cuda.cpu().detach().numpy() == pred.cpu().detach().numpy())
        correct_count += num_correct

        # Backprop
        loss.backward()

        # Update weights
        optim.step()

        # Keep track of losses
        running_loss += loss.item()
    # Calculate average loss
    epoch_loss = running_loss / len(train_loader)

    # Accuracy
    accuracy = correct_count / (len(train_loader) * batch_size)

    losses.append(epoch_loss)
    accuracies.append(accuracy)
    print("Epoch %d - Training loss: %.3f , Training Accuracy: %.3f" %
          (epoch, epoch_loss, accuracy))

    # Validation every 10 epochs
    if epoch % 5 == 0:
        model.eval()
        correct_eval = 0
        eval_loss = 0
        for x,y in val_loader:
            x_cuda = x.cuda()
            y_cuda = torch.squeeze(y).cuda()

            # Output from mode
            output = model(x_cuda)
            output = output.cuda()
            # Loss
            loss = criterion(output, y_cuda)

            # Need to take max over the log probs (batch_size, num_classes)
            _, pred = torch.max(output, 1)
            pred = pred.type(torch.FloatTensor).cuda()

            num_correct = np.sum(y_cuda.cpu().detach().numpy() == pred.cpu().detach().numpy())
            correct_eval += num_correct
            eval_loss += loss.item()
        # Calculate average loss
        epoch_loss = eval_loss / len(val_loader)
        val_loss.append(epoch_loss)

        # Accuracy
        accuracy = correct_eval / (len(val_loader) * batch_size)
        val_accuracy.append(accuracy)
        print("Epoch %d - Validation loss: %.3f , Validation Accuracy: %.3f" %
          (epoch, epoch_loss, accuracy))

# Save training and validation results
# Run info:
print(f'Model: {args.model}')
print(f'Epochs: ', epochs)
print(f'Learning rate: ', lr)
print(f'Weight decay: ', weight_decay)
print(f'Num features: {num_features} + 250 * 128 ')
print(f'Num layers: {args.num_layers}')
print(f'Dropout: {args.dropout}')
path = './Results/' + args.model + f'_{args.run}'
# if not os.path.isdir(path):
#     os.mkdir(path)
Path(path).mkdir(parents=True, exist_ok=True)
    
with open(path+'/hyperparameters.txt', 'w') as file:
    file.write(f'Model: {args.model} \n')
    file.write(f'Epochs: {epochs} \n')
    file.write(f'learning rate: {lr} \n')
    file.write(f'weight decay: {weight_decay} \n')
    file.write(f'Num features: {num_features} + 250 * 128 \n ')
    file.write(f'Num layers: {args.num_layers} \n')
    file.write(f'Dropout: {args.dropout} \n') 

# Train plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(losses)
ax2.plot(accuracies)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax1.set_title('Train loss')
ax2.set_title('Train accuracy')
plt.savefig(path+'/train_plots.png')
plt.show()

# Val plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_loss)
ax2.plot(val_accuracy)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax1.set_title('Val loss')
ax2.set_title('Val accuracy')
plt.savefig(path+'/val_plots.png')
plt.show()

# Save the trained model
torch.save(model.state_dict(), path+'/'+args.model+'_state_dict.pt')

# Run testing on the results
model.eval()
correct_test = 0
test_loss = 0
total_macro = 0
total_micro = 0
for x,y in test_loader:
    x_cuda = x.cuda()
    y_cuda = torch.squeeze(y).cuda()

    # Output from mode
    output = model(x_cuda)
    output = output.cuda()
    
    # Loss
    loss = criterion(output, y_cuda)

    # Need to take max over the log probs (batch_size, num_classes)
    _, pred = torch.max(output, 1)
    pred = pred.type(torch.FloatTensor).cuda()

    num_correct = np.sum(y_cuda.cpu().detach().numpy() == pred.cpu().detach().numpy())
    
    f1 = f1_score(y_cuda.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro')
    total_macro += f1
    f1 = f1_score(y_cuda.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='micro')
    total_micro += f1
    
    correct_test += num_correct
    test_loss += loss
# Calculate average loss
epoch_loss = test_loss / len(test_loader)

# Accuracy
accuracy = correct_test / (len(test_loader) * batch_size)
avg_macro = total_macro / len(test_loader)
avg_micro = total_micro / len(test_loader)
print("Test loss: %.3f , Test Accuracy: %.3f, Avg F1 macro: %.4f, Avg F1 micro: %.3f" % 
      (epoch_loss, accuracy, avg_macro, avg_micro))

# Save test result
with open(path+'/test_results.txt', 'w') as f:
    f.write("Test loss: %.3f , Test Accuracy: %.3f, Avg F1 macro: %.4f, Avg F1 micro: %.3f" % 
      (epoch_loss, accuracy, avg_macro, avg_micro))
