# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:16:51 2019

@author: Mirac
"""


import torch
import pandas as pd
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np


def make_pred_multilabel(data_transforms, model, PATH_TO_MAIN_FOLDER, UNCERTAINTY, N_LABELS, epoch=0):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: the model trained on chexpert images
        PATH_TO_MAIN_FOLDER: path where the extracted chexpert data is located
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 32, can reduce if your GPU has less RAM
    BATCH_SIZE = 32

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)
    

    # create dataloader
    dataset = CheXpertDataset(
        path_to_main_folder=PATH_TO_MAIN_FOLDER,
        fold="test",
        transform=data_transforms['valid'],
        uncertainty=UNCERTAINTY)
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape
        
        if UNCERTAINTY in ["multiclass", 'weighted_multiclass']:
          nn_outputs = model(inputs)
          multiclass_probs = torch.softmax(nn_outputs.view(-1, 3), dim=1)
          outputs = torch.softmax(multiclass_probs[:,[1,2]], dim=1)[:,1].view(-1, N_LABELS)
        else:
          outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if column not in [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']:
                    continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                actual.as_matrix().astype(int), pred.as_matrix())
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)
    
    
    if epoch == 0:
      pred_df.to_csv(os.path.join('results',UNCERTAINTY,'preds.csv'), index=False)
      auc_df.to_csv(os.path.join('results',UNCERTAINTY,'aucs.csv'), index=False)
    
    else:
      pred_df.to_csv(os.path.join('results',UNCERTAINTY,'preds'+ str(epoch) + '.csv'), index=False)
      auc_df.to_csv(os.path.join('results',UNCERTAINTY,'aucs' + str(epoch) + '.csv'), index=False)
    
    return pred_df, auc_df