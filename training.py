import argparse

import torch

from dpdl_defaults import *
import json
import os
from midog_dataset import SlideContainer
import albumentations as A
from midog_dataset import MIDOGTrainDataset
from network import MyRetinaModel
from torchvision.transforms import functional as F
import torch
import numpy as np
import viz_utils
from midog_dataset import MIDOGTrainDataset
from sklearn.model_selection import train_test_split
from midog_dataset import MIDOGTestDataset
import evaluation
import nms
import random

import time
import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)


def get_annotation_data():
    annotation_json = os.path.join(MIDOG_DEFAULT_PATH, MIDOG_JSON_FILE)
    with open(annotation_json) as f:
        annotation_data = json.load(f)

    return annotation_data

def my_sample_func(targets, classes, shape, level_dimensions, level):
    width, height = level_dimensions[level]
    return np.random.randint(0, width - shape[0]), np.random.randint(0, height - shape[1])

def my_improved_sampling_func(targets, classes, shape, level_dimensions, level):
        """
        Samples a random patch from a WSI with probability 0.5, and with probability 0.5 samples a patch that centers around
        a (random) mitotic figure in this region of interest. If the region of interest does not contain any mitotic
        figures, the function should always sample a random patch.

        Parameters:
        targets (list): A list that contains at [0] a list of object bounding boxes, and at [1] a list of classes of these object bounding boxes
        classes (list): List of classes present in the dataset, without background (e.g., [1, 2])
        shape (tuple): Tuple describing the width and height of the patch to be extracted
        level_dimensions (list): List that contains the pixel dimensions per resolution level
        level (int): Resolution level of interest (typically 0)
        """
        # Extract the x and y coordinates of the mitotic figures in the targets
        bboxes, labels = targets
        mitotic_figures = []
        for i in range(len(labels)):
            if labels[i] == 1:
                mitotic_figures.append(bboxes[i])
        hard_figures = []
        for i in range(len(labels)):
            if labels[i] == 2:
                hard_figures.append(bboxes[i])


        mitotic_bool = np.random.choice([True, False], p=[0.5, 0.5])
        if len(mitotic_figures) > 0 and mitotic_bool:
            # Sample a patch of a random mitotic figure
            width, height = level_dimensions[level]
            bbox = random.choice(mitotic_figures)
            x0_mitotic, y0_mitotic, x1_mitotic, y1_mitotic  = bbox

            x_patch = x0_mitotic - shape[0]/2
            y_patch = y0_mitotic - shape[1]/2
            # Ensuring having valid coordinates
            if x_patch < 0:
                x_patch = 0
            if y_patch < 0:
                y_patch = 0
            if x_patch > width - shape[0]:
                x_patch = width - shape[0]
            if y_patch > height - shape[1]:
                y_patch = height - shape[1]
        else:
            # Sample a random patch
            width, height = level_dimensions[level] # width and height of WSI
            x_patch = np.random.randint(0, width - shape[0])
            y_patch = np.random.randint(0, height - shape[1])

        return x_patch, y_patch

def get_cmdline_args_and_run():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', type=str, required=True, help='Execution mode. Options: \'train\', \'val\', \'test\'')
    parser.add_argument('-p', '--patchsize', type=int, default=512, help='Patchsize - network will use pxp patches during training and inference')
    parser.add_argument('-b', '--batchsize', type=int, default=12, help='Batchsize')
    parser.add_argument('-nt', '--npatchtrain', type=int, default=10, help='Number of patches per slide during training')
    parser.add_argument('-nv', '--npatchval', type=int, default=10, help='Number of patches per slide during validation')
    parser.add_argument('-ne', '--nepochs', type=int, default=200, help='Total number of epochs for training')
    parser.add_argument('-se', '--startepoch', type=int, default=0, help='Starting epoch for training (remaining number of training epochs is nepochs-startepoch)')
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--logdir', type=str, default='./output/', help='Directory for lightning logs/checkpoints')
    # parser.add_argument('--resdir', type=str, default='./results', help='Directory for result files')
    parser.add_argument('-c', '--checkptfile', type=str, default=None, help='Path to model file (necessary for reloading/retraining)')
    parser.add_argument('-s', '--seed', type=int, default='31415', help='Seed for randomness, default=31415; set to -1 for random')

    args = parser.parse_args()

    possible_execution_modes = ['train', 'val', 'test']
    if not args.mode in possible_execution_modes:
        print('Error: Execution mode {} is unknown. Please choose one of {}'.format(args.mode, possible_execution_modes))

    if not args.seed == -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.logdir):
        print("Creating log directory {}".format(args.logdir))
        os.makedirs(args.logdir)

    if args.mode == 'train':
        training_val(args)

    if args.mode == 'val':
        training_val(args)

    if args.mode == 'test':
        test(args)

def training_val(args):
    annotation_data = get_annotation_data()

    # TODO filter
    list_image_filenames = TRAINING_IDS

    # TODO: Normalization
    tfms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.1, p=0.5),
                #A.Normalize(mean=[0.5, 0.58, 0.65], std=[0.2, 0.225, 0.25])],
                ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # Note: we only get the mitotic figures, not the impostors
    categories = [1]
    res_level = 0

    containers = []
    for image_filename in list_image_filenames:
        # filter annotations by image_id and desired annotation type
        bboxes, labels = viz_utils.get_bboxes(image_filename, annotation_data, categories)

        image_id = viz_utils.image_filename2id(image_filename)
        containers.append(SlideContainer(os.path.join(MIDOG_DEFAULT_PATH, image_filename), image_id, y=[bboxes, labels],
                                        level=res_level, width=args.patchsize, height=args.patchsize))

    # split the train set into train and validation
    train_containers, val_containers = train_test_split(containers, train_size=0.8, random_state=args.seed)

    dataset = MIDOGTrainDataset(train_containers, patches_per_slide=args.npatchtrain, transform=tfms, sample_func=my_improved_sampling_func)

    val_dataset = MIDOGTrainDataset(val_containers, patches_per_slide=args.npatchval, transform=tfms, sample_func=my_sample_func)

    # this is not ideal but use num_workers=0 - there seems to be an bug in openslide that causes missed pixels during multi-threading
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchsize, shuffle=True, num_workers=0, collate_fn=viz_utils.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0, collate_fn=viz_utils.collate_fn)

    print("Cuda available: {}".format(torch.cuda.is_available()), flush=True)
    # Initialize a trainer
    cur_time = time.time()
    time_str = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d-%H-%M-%S')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        logger = TensorBoardLogger(save_dir=args.logdir, version='version_lr{lr}_p{p}_b{b}_{t}'.format(
        lr=args.learningrate, p=args.patchsize, b=args.batchsize, t=time_str)),
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=args.nepochs,
        log_every_n_steps=5,
        callbacks = [lr_monitor]
    )
    my_detection_model = MyRetinaModel(num_classes=2, iterations_epoch=len(data_loader), lr=args.learningrate, epochs=args.nepochs)
    # Train the model

    ckpt = None
    if args.checkptfile:
        ckpt = args.checkptfile
        my_detection_model = MyRetinaModel.load_from_checkpoint(ckpt)
        print("Model saved!")
    if args.mode == 'train':
        trainer.fit(my_detection_model, data_loader, val_data_loader, ckpt_path=ckpt)
    if args.mode == 'val':
        trainer.validate(my_detection_model, val_data_loader, ckpt_path=ckpt)


def test(args):
    annotation_data = get_annotation_data()
    categories = [1]
    slide_folder = MIDOG_DEFAULT_PATH

    # TODO
    list_test_image_filenames = TEST_IDS

    test_datasets = []
    test_batchsize = args.batchsize
    categories = [1]
    res_level= 0

    all_predictions = []
    all_gt = []

    # Initialize a trainer
    cur_time = time.time()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        #logger = TensorBoardLogger(save_dir=args.logdir, version='version_lr{lr}_p{p}_b_{b}_{t}'.format(
        # lr=args.learningrate, p=args.patchsize, b=args.batchsize, t=cur_time)),
        logger = None,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=args.nepochs,
        log_every_n_steps=5,
        callbacks = [lr_monitor]
    )

    ckpt = args.checkptfile
    my_detection_model = MyRetinaModel.load_from_checkpoint(ckpt)

    for image_filename in list_test_image_filenames:
        # get the corresponding container and dataset
        image_id = viz_utils.image_filename2id(image_filename)
        bboxes, labels = viz_utils.get_bboxes(image_filename, annotation_data, categories)
        container = SlideContainer(os.path.join(slide_folder, image_filename), image_id, y=[bboxes, labels],
                                level=res_level, width=args.patchsize, height=args.patchsize)

        # in this case we want to have one slide per dataset/data loader --> iterate over each slide separately to avoid confusion
        cur_dataset = MIDOGTestDataset(container)
        cur_test_dataloader = torch.utils.data.DataLoader(cur_dataset, batch_size=test_batchsize,
                                                          shuffle=False, # important!
                                                          num_workers=0, collate_fn=viz_utils.collate_fn)

        prediction = trainer.predict(model=my_detection_model, dataloaders=cur_test_dataloader)
        image_pred = torch.empty((0, 4))
        image_scores_raw = torch.empty((0))

        # format of prediction:[batches] --> batch: (tuple of x, tuple of pred(x)) --> pred(x): {dict: boxes/pred}
        for batch_id, pred_batch in enumerate(prediction):
            for image_id, pred in enumerate(pred_batch[1]):
                # transform prediction to global coordinates
                cur_global_pred = cur_dataset.local_to_global(batch_id * test_batchsize + image_id, pred['boxes'])
                image_pred = torch.cat([image_pred, cur_global_pred]) # me: ASK why is there concatination here? is there a skip connection here?
                image_scores_raw = torch.cat([image_scores_raw, pred["scores"]])

        # can also be handled within the network
        image_pred_th = image_pred[image_scores_raw > 0.5] # me: confident predictions
        image_pred_cthw = viz_utils.tlbr2cthw(image_pred_th)[:, :2] # me: convertsfrom the "top-left bottom-right" (TLBR) format to the "center-top width-height" (CTHW) with the width and height values removed.
        image_pred_cthw = nms.nms(image_pred_cthw, 0.4) # me: contains only the non-overlapping bounding boxes with the highest scores.
        image_gt_cthw = viz_utils.tlbr2cthw(cur_dataset.get_slide_labels_as_dict()['boxes'])[:, :2] # me: converts from TLBR to CTHW format, and only the center-top coordinates are kept

        all_predictions.append(image_pred_cthw)
        all_gt.append(image_gt_cthw)

    # Final evaluation on the test set
    tp, fp, fn = evaluation.get_confusion_matrix(all_gt, all_predictions)
    aggregates = evaluation.get_metrics(tp, fp, fn)
    print("The performance on the test set for the current setting was \n"+
        "F1-score:  {:.3f}\n".format(aggregates["f1_score"]) +
        "Precision: {:.3f}\n".format(aggregates["precision"]) +
        "Recall:    {:.3f}\n".format(aggregates["recall"]))


if __name__ == "__main__":
    get_cmdline_args_and_run()
