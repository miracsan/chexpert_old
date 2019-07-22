# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:16:02 2019

@author: Mirac
"""

def train_cnn(PATH_TO_MAIN_FOLDER, LR, WEIGHT_DECAY, UNCERTAINTY="zeros", USE_MODEL=0):
    """
    Train a model with chexpert data using the given hyperparameters

    Args:
        PATH_TO_MAIN_FOLDER: path where the extracted chexpert data is located
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD
        UNCERTAINTY: the uncertainty method to be used in training
        USE_MODEL: specify the checkpoint object if you want to continue 
            training  

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 8
    BATCH_SIZE = 32
    
    
    if USE_MODEL == 0:      
      try:
          rmtree(os.path.join('results',UNCERTAINTY))
      except BaseException:
          pass  # directory doesn't yet exist, no need to clear it
      os.makedirs(os.path.join('results',UNCERTAINTY))
    
    
    
    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels


    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CheXpertDataset(
        path_to_main_folder=PATH_TO_MAIN_FOLDER,
        fold='train',
        transform=data_transforms['train'],
        uncertainty=UNCERTAINTY)
    transformed_datasets['valid'] = CheXpertDataset(
        path_to_main_folder=PATH_TO_MAIN_FOLDER,
        fold='valid',
        transform=data_transforms['valid'],
        uncertainty=UNCERTAINTY)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        transformed_datasets['valid'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    
    
    if not USE_MODEL == 0:
      model = USE_MODEL['model']
      starting_epoch = USE_MODEL['epoch']
    else:
      starting_epoch = 0
      model = models.densenet121(pretrained=True)
      num_ftrs = model.classifier.in_features
      # add final layer with # outputs in same dimension of labels with sigmoid
      # activation
      if UNCERTAINTY in ["multiclass", 'weighted_multiclass']:
          model.classifier = nn.Sequential(
          nn.Linear(num_ftrs, 3 * N_LABELS), nn.Sigmoid())
      else:
          model.classifier = nn.Sequential(
          nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())


    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    if UNCERTAINTY == "ignore":
        criterion = BCEwithIgnore()
    elif UNCERTAINTY == "multiclass":
        criterion = nn.CrossEntropyLoss()
    elif UNCERTAINTY == 'weighted_multiclass':
        label_weights = torch.tensor(transformed_datasets['train'].getWeights(uncertainty='weighted_multiclass'))
        label_weights = label_weights.to(torch.device("cuda"))
        criterion = WeightedCrossEntropy(label_weights)
    elif UNCERTAINTY == "weighted_zeros":
        label_weights = torch.tensor(transformed_datasets['train'].getWeights())
        label_weights = label_weights.to(torch.device("cuda"))
        criterion = WeightedBCE(label_weights)
    else:
        criterion = nn.BCELoss()
        
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'valid']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,
                                    uncertainty=UNCERTAINTY,
                                    starting_epoch=starting_epoch)

    # get preds and AUCs on test fold
    preds, aucs = make_pred_multilabel(data_transforms,
                                       model,
                                       PATH_TO_MAIN_FOLDER,
                                       UNCERTAINTY,
                                       N_LABELS)

    return preds, aucs
