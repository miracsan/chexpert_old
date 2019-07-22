# chexpert
This project was done as part of the Machine Learning in Medical Imaging practical course at the Technical University of Munich,
  summer 2019.
  
Here I tried to reproduce the results in the CheXpert paper: https://arxiv.org/abs/1901.07031

Model training, predictions and class activation maps are all implemented.

The four uncertainty handling methods described in the paper ("zeros", "ones", "ignore", "multiclass") are available. In addition, 
there are the weighted cross entropy versions of zeros and multiclass under the names "weighted_zeros" and "weighted_multiclass" 

All the work was done on Google Colab, where the CheXpert dataset was also stored. To run the code, upload the dataset (available from
https://stanfordmlgroup.github.io/competitions/chexpert/) to your Google Drive and follow the instructions on the MainColab.ipynb file.

In my implementation, I got results which were very close to the paper: (Here I report only the average of the 5 pathologies in the paper)
  For zeros: 86.2% vs 88.9% in the paper
  For ones: 86.4% vs 89.3% in the paper
  For ignore: 85.8% vs 88.9% in the paper
  For multiclass: 86.2% vs 89.5% in the paper

  
