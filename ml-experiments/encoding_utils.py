"""
encoding_utils.py: Houses common classes/functions used across all experiments/models
"""

from transformers import (
    AutoTokenizer,
    AutoModel,
    pipeline,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
import datasets
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from typing import List
import os
import errno
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
from datetime import datetime, time
import torchmetrics
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import encoding_utils
import re
from nltk.corpus import stopwords
import string
import seaborn as sns
import pickle
from joblib import Parallel, delayed
from typing import Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.base import ClassifierMixin

STOPWORDS = set(stopwords.words("english"))
PUNCT_TO_REMOVE = string.punctuation

######################
### Util Functions ###
######################


def temporal_processing(
    df: pd.DataFrame, discharge: bool = True, notes: bool = True, cutoff_visits: int = 0
):
    """Perform time-relevant processing, as well as standardizing the time records

    Parameters:
        df (pd.DataFrame): Dataframe to be edited with regards to the imageslabels
        cutoff_visits (int): Minimum number of visits associated with each x-ray label
        discharge (bool): True if we wish to filter to entries with a discharge time before the x-ray
        notes (bool): True if we wish to filter to entries with a note charttime before the x-ray

    Returns: pd.DataFrame processed to our liking

    """
    # Fix the timestamp of the X-Ray metadata
    df["StudyDate"] = (
        pd.to_datetime(df["StudyDate"].astype(str), format="%Y%m%d")
        .dt.strftime("%Y-%m-%d")
        .astype(str)
    )
    df["StudyTime"] = pd.to_datetime(
        df["StudyTime"].apply(
            lambda x: f"{int(x):06}.{int(round((x - int(x)) * 1000)):03}"
        ),
        format="%H%M%S.%f",
    ).dt.time.astype(str)
    df["studyts"] = df.apply(
        lambda row: f"{row['StudyDate']} {datetime.strptime(row['StudyTime'], '%H:%M:%S.%f').strftime('%H:%M:%S.%f').rstrip('0').rstrip('.')}"
        if "." in row["StudyTime"]
        else f"{row['StudyDate']} {datetime.strptime(row['StudyTime'], '%H:%M:%S').strftime('%H:%M:%S')}",
        axis=1,
    )
    df["studyts"] = df["studyts"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
        if "." in x
        else datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )

    # Remove X-Ray studies that occured before an admission discharge
    if discharge:
        df = df[df["studyts"] > df["dischtime"]]
    # Remove X-Ray studies that occured before a note was registered
    if notes:
        df = df[df["studyts"] > df["charttime"]]
    if cutoff_visits >= 2:
        num_visits = df.groupby("subject_id")["admittime"].nunique()
        num_visits = num_visits[num_visits >= cutoff_visits]
        subjects_to_keep = num_visits.index.tolist()
        df = df[df["subject_id"].isin(subjects_to_keep)]
    return df


def label_processing(df: pd.DataFrame, labels: list[str] = []):
    """Convert labels to 0/1 scheme, and leave NA values for later.

    Parameters:
        df (pd.DataFrame): Dataframe to be edited with regards to the labels
        labels (list[str]): List of labels that we would like to process

    Returns: pd.DataFrame processed to our liking,

    """
    for label in labels:
        df.loc[df[label] == -1, label] = 0
    return df


def process_data(
    df: pd.DataFrame,
    labels: list[str] = [],
    cutoff_visits: int = 0,
    discharge: bool = True,
    notes: bool = True,
):
    """Preprocess input dataframe by only retaining entries/notes occuring before the first X-Ray

    Parameters:
        df (pd.DataFrame): Raw dataframe
        labels (list[str]): List of labels that we would like to process
        cutoff_visits (int): Minimum number of visits associated with each x-ray label
        discharge (bool): True if we wish to filter to entries with a discharge time before the x-ray
        notes (bool): True if we wish to filter to entries with a note charttime before the x-ray

    Returns: pd.DataFrame, processed to our liking

    """
    # process time-relevant aspects of data if necessary:
    if (discharge) or (cutoff_visits >= 2) or (notes):
        df = temporal_processing(df, discharge, notes, cutoff_visits)
    # label processing
    df = label_processing(df, labels)
    return df


def create_train_df(df: pd.DataFrame, labels: list[str] = []):
    """Function to generate dataframe to tokenize

    Parameters:
        df (str): raw data that has been processed
        labels (list[str]): List of labels that we would like to process

    Returns: Dataframe subsetted to entries with notes occuring before the first X-Ray

    """
    print("Subsetting to notes that occured before the X-Ray\n")
    # read in the data
    df_processed = process_data(df, labels, cutoff_visits=0, discharge=True, notes=True)
    df_final = df_processed.drop_duplicates()
    # check that dropping the duplicates didn't change our data
    assert df_processed["subject_id"].nunique() == df_final["subject_id"].nunique()
    assert df_processed["hadm_id"].nunique() == df_final["hadm_id"].nunique()
    assert df_processed["admittime"].nunique() == df_final["admittime"].nunique()
    assert df_processed["dischtime"].nunique() == df_final["dischtime"].nunique()
    assert df_processed["study_id"].nunique() == df_final["study_id"].nunique()
    assert df_processed["studyts"].nunique() == df_final["studyts"].nunique()
    assert df_processed["charttime"].nunique() == df_final["charttime"].nunique()
    print("Finished processing data\n")
    return df_final


def vision_dataloaders(
    df: pd.DataFrame,
    label_columns: list[str],
    transforms: torchvision.transforms.Compose,
    batch_size: int = 128,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    shuffle: bool = False,
):
    """Prepare and return a DataLoader for vision-based tasks with custom transformations.

    Parameters:
        df (pd.DataFrame): Dataframe containing paths to images and associated labels.
        label_columns (list): List of column names representing labels in the dataframe.
        transforms (torchvision.transforms): Transformations to apply to each image.
        batch_size (int, optional): Number of images per batch. Default is 128.
        rank (int, optional): Rank of the process (for distributed training). Default is None.
        world_size (int, optional): Total number of processes (for distributed training). Default is None.
        shuffle (bool, optional): Whether to shuffle the data. Default is False.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    dataset = CustomImageDataset(
        # label_columns is a list containing the names of the labels
        labels=df[label_columns].values,
        img_path=df["images"].values,
        transform=transforms,
    )
    if world_size == 1:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    return dataloader


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    rank: int,
    batch_size: int,
    max_grad_norm: float,
):
    """Train the model for one epoch, calculate accuracy and loss, and update AUROC metrics.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader with the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function for calculating error.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        rank (int): Device rank for distributed training.
        batch_size (int): Batch size used for training.
        max_grad_norm (float): Maximum norm for gradient clipping.

    Returns:
        tuple: Training accuracy, training loss, and AUROC score.
    """
    model.train()
    gc.collect()
    # Initialize BinaryAUROC metric
    auroc_metric = torchmetrics.classification.MultilabelAUROC(
        num_labels=14, average="none"
    ).to(rank)
    # Progress Bar
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Train",
        ncols=5,
    )
    num_correct = 0
    total_loss = 0
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients
        images, labels = images.to(rank), labels.to(rank)
        outputs = model(images)
        loss = criterion(outputs, labels.to(torch.float32))
        # Update AUROC metric
        auroc_metric.update(outputs, labels)
        # Update no. of correct predictions & loss as we iterate
        num_correct += int((outputs.round().squeeze() == labels).sum())
        total_loss += float(loss.item())
        auroc = auroc_metric.compute()
        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (14 * batch_size * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
            auroc=auroc,
        )
        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()
        batch_bar.update()  # Update tqdm bar
        ### Efforts to resolve CUDA OOM errors ###
        del images
        del labels
        del loss
        ### Efforts to resolve CUDA OOM errors ###

    batch_bar.close()  # You need this to close the tqdm bar
    acc = 100 * num_correct / (14 * batch_size * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    # Reset the AUROC metric for the next epoch
    auroc_metric.reset()
    return acc, total_loss, auroc


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    rank: int,
    batch_size: int,
):
    """Evaluate the model on the validation set and calculate accuracy, loss, and AUROC.

    Parameters:
        model (torch.nn.Module): The model to be validated.
        dataloader (DataLoader): DataLoader with the validation data.
        criterion (torch.nn.Module): Loss function for calculating error.
        rank (int): Device rank for distributed validation.
        batch_size (int): Batch size used for validation.

    Returns:
        tuple: Validation accuracy, validation loss, and AUROC score.
    """
    model.eval()
    # Initialize BinaryAUROC metric
    auroc_metric = torchmetrics.classification.MultilabelAUROC(
        num_labels=14, average="none"
    ).to(rank)
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Val",
        ncols=5,
    )
    num_correct = 0.0
    total_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        # Move images to device
        images, labels = images.to(rank), labels.to(rank)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels.to(torch.float32))
        # Update AUROC metric
        auroc_metric.update(outputs, labels)
        num_correct += int((outputs.round().squeeze() == labels).sum())
        total_loss += float(loss.item())
        auroc = auroc_metric.compute()
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (14 * batch_size * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            auroc=auroc,
        )
        batch_bar.update()
        del images
        del labels
        del loss
    batch_bar.close()
    acc = 100 * num_correct / (14 * batch_size * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    # Reset the AUROC metric for the next epoch
    auroc_metric.reset()
    return acc, total_loss, auroc


def cleanup():
    """Clean up the distributed training process by destroying the process group."""
    dist.destroy_process_group()


def setup(rank: int, world_size: int):
    """Set up the distributed environment for multi-GPU training.

    Parameters:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes for distributed training.

    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Initialize the process group for DDP
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_training(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    max_grad_norm: float,
    label_columns: list[str],
    epochs: int,
):
    """Coordinate distributed training across multiple GPUs, using DataParallel.

    Parameters:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        model (torch.nn.Module): Model to be trained.
        train_df (pd.DataFrame): Dataframe containing training data.
        val_df (pd.DataFrame): Dataframe containing validation data.
        batch_size (int): Batch size for each GPU.
        max_grad_norm (float): Maximum gradient norm for clipping.
        label_columns (list): List of column names for labels in the dataframes.
        epochs (int): Number of training epochs.

    Returns:
        None
    """
    setup(rank, world_size)
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    valid_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    # Set up DDP model
    torch.cuda.set_device(rank)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    train_dataloader = vision_dataloaders(
        train_df, label_columns, train_transforms, batch_size, rank, world_size
    )
    val_dataloader = vision_dataloaders(
        val_df, label_columns, valid_transforms, batch_size, rank, world_size
    )
    # Define optimizer, criterion, and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.BCELoss().to(rank)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    best_trainacc = 0.0
    best_valloss = 999.0
    best_epoch = 0
    for epoch in range(epochs):
        curr_lr = float(optimizer.param_groups[0]["lr"])
        train_acc, train_loss, train_auroc = train(
            model,
            train_dataloader,
            optimizer,
            criterion,
            scaler,
            rank,
            batch_size,
            max_grad_norm,
        )
        print(
            "\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Train AUROC {}\t Learning Rate {:.04f}".format(
                epoch + 1, epochs, train_acc, train_loss, train_auroc, curr_lr
            )
        )
        if train_acc >= best_trainacc:
            best_trainacc = train_acc
        val_acc, val_loss, val_auroc = validate(
            model, val_dataloader, criterion, rank, batch_size
        )
        print(
            "Val Acc {:.04f}%\t Val Loss {:.04f}\t Val AUROC {}".format(
                val_acc, val_loss, val_auroc
            )
        )
        scheduler.step(val_acc)
        if val_loss <= best_valloss:
            best_valloss = val_loss
            best_epoch = epoch
            if rank == 0:  # Ensure only one process saves the model
                torch.save(model.state_dict(), "densenet121_vision/best_model.pth")
                print(f"Best model saved with Val Loss: {best_valloss:.04f}%")
        if best_epoch - epoch >= 3:
            print("Early stopping")
            break
    cleanup()


def chunk(tokenizer, note: str):
    """Function to iteratively generate non-overlapping tokens for medical texts

    Parameters:
        tokenizer (Tokenizer): tokenizer associated with the model of interest
        note (str): text to be tokenized

    Returns: chunked_input_ids (torch.Tensor): nested input_ids into the model
             chunked_attention_mask (torch.Tensor): nested attention_masks into the model

    """
    split_note = note.split("eotextdelimiter")
    tokens = list(
        map(
            lambda n: tokenizer(
                n, truncation=False, add_special_tokens=False, padding="max_length"
            ),
            split_note,
        )
    )

    chunked_input_ids = [
        torch.tensor(tokens[i]["input_ids"]) for i in range(len(tokens))
    ]
    chunked_attention_mask = [
        torch.tensor(tokens[i]["attention_mask"]) for i in range(len(tokens))
    ]

    return (chunked_input_ids, chunked_attention_mask)


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans("", "", PUNCT_TO_REMOVE))


def text_preprocessing(note):
    """Function to preprocess medical text for embeddings. Currently, implemented punctuation removal,
        stopword removal, replacement of repetitive words, removal of `x95` character, and lower-casing.

    Parameters:
        note (pandas.core.series.Series): text to be processed

    Returns:
        note (pandas.core.series.Series): processed text

    """
    # Punctuation removal
    note = note.apply(lambda text: remove_punctuation(text))
    # Stopword removal
    note = note.apply(lambda text: remove_stopwords(text))
    # Frequent/repetitive word removal
    header = "Name Unit No Admission Date Discharge Date Date Birth "
    note = note.apply(lambda x: x.replace(header, ""))
    # `\x95 character removal
    note = note.apply(lambda x: re.sub(r"\x95", "", x))
    # Lower-casing
    note = note.str.lower()
    return note


def train_mlp_classifier(
    dataset: pd.DataFrame,
    label_name: str,
    labels: np.ndarray,
    sparse_mat_pcode: np.ndarray,
    output_dir: str,
):
    """
    Training function for an MLP classifier on a specific label. Saves performance metrics, calibration plots, and
    confidence levels to specified output paths.

    Parameters:
    - unique_subjects (pd.DataFrame): Raw data to split
    - label_name (str): The label name to train the classifier on.
    - labels (np.ndarray): Array of binary labels indicating presence or absence of the label.
    - sparse_mat_pcode (np.ndarray): Sparse matrix of feature representations for each instance.
    - output_dir (str): Directory to save metrics, calibration plots, and confidence levels.

    Returns:
    - dict: Dictionary with confidence levels as keys and indices of confident predictions as values.
    """
    # Define output file paths
    metrics_csv_path = os.path.join(
        output_dir, f"metrics_label_{label_name}_clf_MLP.csv"
    )
    calibration_plot_path = os.path.join(
        output_dir, f"calibration_plot_{label_name}_clf_MLP.pdf"
    )
    model_path = f"text_models/{label_name}.pkl"
    confident_predictions_path = f"confident_predictions/{label_name}.pkl"

    # Splitting the data into training and testing sets
    ##    # Get unique subject IDs
    ##    unique_subjects = dataset['subject_id'].unique()
    ##    # Split subject IDs into train/test groups
    ##    train_subjects, test_subjects = train_test_split(
    ##        unique_subjects, test_size=0.2, random_state=42
    ##    )
    ##
    ##    # Filter dataset to get rows for those subjects
    ##    df_train = dataset[dataset['subject_id'].isin(train_subjects)]
    ##    df_test  = dataset[dataset['subject_id'].isin(test_subjects)]
    ##
    ##    # Extract features and labels
    ##    X_tr = sparse_mat_pcode[df_train.index]
    ##    X_calib = sparse_mat_pcode[df_test.index]
    ##    Y_tr = labels[df_train.index]
    ##    Y_calib = labels[df_test.index]

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        sparse_mat_pcode, labels, test_size=0.2, random_state=42
    )

    # Check if model already exists and load it
    if os.path.exists(model_path):
        print(f"Loading existing model for {label_name} from {model_path}")
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
    else:
        print(f"Training and calibrating MLP on label {label_name}...\n")

        # Define the MLP classifier
        clf = MLPClassifier(
            early_stopping=True, validation_fraction=0.1, max_iter=300, random_state=42
        )

        print("Training MLP base classifier...")
        clf.fit(sparse_mat_pcode, labels)

        # Save the trained model
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

    # Scoring the calibrated classifier
    y_prob_calibrated = clf.predict_proba(X_te)[:, 1]
    auroc = metrics.roc_auc_score(labels, y_prob_calibrated)

    # Calibration curve data
    prob_true, prob_pred = calibration_curve(Y_te, y_prob_calibrated, n_bins=10)

    # Plotting the calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibrated MLP", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Plot for {label_name}")
    plt.legend()
    plt.grid()
    plt.savefig(calibration_plot_path)
    plt.close()

    # Get confident predictions for confidence levels from 0.01 to 0.05 (in 0.01 steps) and from 0.05 to 0.45 (in 0.05 steps)
    y_prob_calibrated_all = clf.predict_proba(sparse_mat_pcode)[:, 1]
    confident_indices_dict = {}

    # Confidence levels from 0.01 to 0.05 (in 0.01 increments)
    for confidence_level in np.arange(0.01, 0.06, 0.01):
        confident_indices = np.where(
            (y_prob_calibrated_all < confidence_level)
            | (y_prob_calibrated_all > (1 - confidence_level))
        )[0]
        confident_indices_dict[confidence_level] = confident_indices
        print(
            f"Confidence Level {confidence_level:.2f}: Number of confident predictions: {len(confident_indices)}"
        )

    # Confidence levels from 0.05 to 0.45 (in 0.05 increments)
    for confidence_level in np.arange(0.05, 0.50, 0.05):
        confident_indices = np.where(
            (y_prob_calibrated_all < confidence_level)
            | (y_prob_calibrated_all > (1 - confidence_level))
        )[0]
        confident_indices_dict[confidence_level] = confident_indices
        print(
            f"Confidence Level {confidence_level:.2f}: Number of confident predictions: {len(confident_indices)}"
        )

    # Save confident predictions
    with open(confident_predictions_path, "wb") as f:
        pickle.dump(confident_indices_dict, f)

    # Save performance metrics
    metrics_result = {
        "label_name": label_name,
        "classifier_name": "MLP",
        "AUROC": auroc,
    }
    metrics_df = pd.DataFrame([metrics_result])
    metrics_df.to_csv(metrics_csv_path, index=False)

    print(
        f"Finished training and calibrating MLP on label {label_name}. Metrics saved."
    )

    return confident_indices_dict


from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin
import numpy as np
import os
import pickle


def evaluate_classifier(
    model_path: str,
    label_name: str,
    labels: np.ndarray,
    embeddings: np.ndarray,
    output_dir: str,
    mode="test",
):
    """
    Evaluates a pre-trained classifier on test data, computing metrics and generating plots.
    Handles various classifier types uniformly.

    Parameters:
    - model_path (str): Path to the saved model
    - label_name (str): The label name being evaluated
    - labels (np.ndarray): Array of binary labels
    - embeddings (np.ndarray): Feature embeddings for test data
    - output_dir (str): Directory to save outputs
    - mode (str): 'test' or other identifier for filenames
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("confident_predictions_test/", exist_ok=True)
    # Define output paths
    metrics_csv_path = os.path.join(
        output_dir, f"metrics_label_{label_name}_{mode}.csv"
    )
    calibration_plot_path = os.path.join(
        output_dir, f"calibration_plot_{label_name}_{mode}.pdf"
    )
    confident_predictions_path = f"confident_predictions_test/{label_name}.pkl"
    # Load pre-trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    if not isinstance(clf, ClassifierMixin):
        raise ValueError(
            f"Loaded object from {model_path} is not a scikit-learn classifier"
        )
    # Handle classifiers that don't support predict_proba natively
    if not hasattr(clf, "predict_proba"):
        # Calibrate classifiers like LinearSVC that need probability estimation
        calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        try:
            calibrated_clf.fit(embeddings, labels)  # Needs some data to calibrate
            y_prob = calibrated_clf.predict_proba(embeddings)[:, 1]
        except Exception as e:
            raise RuntimeError(f"Probability calibration failed: {str(e)}")
    else:
        # For classifiers with native predict_proba
        y_prob = clf.predict_proba(embeddings)[:, 1]
    # Get confident predictions at various thresholds
    confident_indices_dict = {}
    # Confidence levels from 0.01 to 0.05 (in 0.01 increments)
    for confidence_level in np.arange(0.01, 0.06, 0.01):
        confident_indices = np.where(
            (y_prob < confidence_level) | (y_prob > (1 - confidence_level))
        )[0]
        confident_indices_dict[confidence_level] = confident_indices
        print(
            f"Confidence Level {confidence_level:.2f}: Number of confident predictions: {len(confident_indices)}"
        )
    # Confidence levels from 0.05 to 0.45 (in 0.05 increments)
    for confidence_level in np.arange(0.05, 0.50, 0.05):
        confident_indices = np.where(
            (y_prob < confidence_level) | (y_prob > (1 - confidence_level))
        )[0]
        confident_indices_dict[confidence_level] = confident_indices
        print(
            f"Confidence Level {confidence_level:.2f}: Number of confident predictions: {len(confident_indices)}"
        )
    # Save confident predictions
    with open(confident_predictions_path, "wb") as f:
        pickle.dump(confident_indices_dict, f)
    print(
        f"Finished evaluating {clf.__class__.__name__} on label {label_name} ({mode} set)"
    )
    return confident_indices_dict


def remove_module_prefix(state_dict: dict):
    """
    Remove 'module.' prefix from keys in the state dictionary to simplify parameter names.

    Parameters:
    - state_dict (dict): State dictionary from a model checkpoint, potentially containing 'module.' prefixes in keys.

    Returns:
    - dict: State dictionary with 'module.' prefix removed from keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def aggregate_results(output_dir: str):
    """
    Aggregates individual CSV files in the output directory into a single CSV for AUROC and other metrics.

    Parameters:
    - output_dir (str): Directory containing individual CSV files with AUROC and other metric results.

    Saves:
    - A single CSV file with aggregated metrics.
    """
    metrics_data = []

    # Iterate over all files in the directory
    for filename in os.listdir(output_dir):
        if filename.endswith(".csv") and filename.startswith("metrics_label"):
            # Extract label name from filename
            parts = filename.split("_")
            label_name = parts[2]

            metrics_csv_path = os.path.join(output_dir, filename)
            metrics_df = pd.read_csv(metrics_csv_path)

            metrics_row = {
                "Label": label_name,
                "MLP_AUROC": metrics_df["AUROC"].values[0],
            }
            metrics_data.append(metrics_row)

    # Save aggregated results
    metrics_aggregated_df = pd.DataFrame(metrics_data)
    metrics_aggregated_csv_path = os.path.join(
        output_dir, "aggregated_metrics_results.csv"
    )
    metrics_aggregated_df.to_csv(metrics_aggregated_csv_path, index=False)


def bootstrap_auroc(
    label_index: int,
    confidence_level: float,
    non_confident_indices: np.ndarray,
    num_bootstraps: int,
    all_outputs: np.ndarray,
    all_labels: np.ndarray,
):
    """
    Perform bootstrapping to estimate AUROC and confidence intervals.

    Parameters:
    - label_index (int): Index of the label being evaluated.
    - confidence_level (float): Confidence level threshold for selecting non-confident indices.
    - non_confident_indices (np.ndarray): Indices of samples deemed non-confident for bootstrapping.
    - num_bootstraps (int): Number of bootstrap samples to draw.
    - all_outputs (np.ndarray): Array of model outputs.
    - all_labels (np.ndarray): Array of true labels.

    Returns:
    - tuple: Mean AUROC and a confidence interval array [CI_low, CI_high].
    """
    bootstrapped_aurocs = []
    # Perform the bootstrapping
    for j in tqdm(range(num_bootstraps), desc="Bootstrapping", unit="iter"):
        auroc_metric = torchmetrics.classification.MultilabelAUROC(
            num_labels=14, average="none"
        )
        # Perform stratified sampling on non-confident indices
        boot_outputs, boot_labels = resample(
            all_outputs[non_confident_indices],
            all_labels[non_confident_indices],
            stratify=all_labels[non_confident_indices],
        )
        # Calculate AUROC for the bootstrap sample
        auroc_metric.update(boot_outputs, boot_labels)
        auroc = auroc_metric.compute()
        bootstrapped_aurocs.append(auroc)
        # print(j)
    # Calculate mean AUROC and confidence intervals
    stacked_aurocs = torch.stack(bootstrapped_aurocs, dim=0)
    mean_auroc = torch.mean(stacked_aurocs, dim=0)
    lower_percentile = np.percentile(stacked_aurocs, 2.5, axis=0)
    upper_percentile = np.percentile(stacked_aurocs, 97.5, axis=0)
    auroc_ci = np.array([lower_percentile, upper_percentile])
    return mean_auroc, auroc_ci


def bootstrap_auroc_difference_threshold(
    label_index: int,
    confidence_level: float,
    confident_indices: np.ndarray,
    non_confident_indices: np.ndarray,
    num_bootstraps: int,
    all_outputs: np.ndarray,
    all_labels: np.ndarray,
):
    """
    Perform bootstrapping to estimate AUROC metrics for both subsets and their difference.

    Returns:
    - tuple: (positive_mean, positive_ci, negative_mean, negative_ci, diff_mean, diff_ci)
    """
    bootstrapped_pos = []
    bootstrapped_neg = []
    bootstrapped_diffs = []

    for _ in tqdm(range(num_bootstraps), desc="Bootstrapping", unit="iter"):
        auroc_metric = torchmetrics.classification.MultilabelAUROC(
            num_labels=14, average="none"
        )

        # Resample both subsets
        boot_outputs_conf, boot_labels_conf = resample(
            all_outputs[confident_indices],
            all_labels[confident_indices],
            stratify=all_labels[confident_indices],
        )
        boot_outputs_nonconf, boot_labels_nonconf = resample(
            all_outputs[non_confident_indices],
            all_labels[non_confident_indices],
            stratify=all_labels[non_confident_indices],
        )

        # Compute AUROCs
        auroc_metric.update(boot_outputs_conf, boot_labels_conf)
        auroc_conf = auroc_metric.compute()[label_index]
        auroc_metric.reset()

        auroc_metric.update(boot_outputs_nonconf, boot_labels_nonconf)
        auroc_nonconf = auroc_metric.compute()[label_index]

        # Store results
        bootstrapped_pos.append(auroc_conf)
        bootstrapped_neg.append(auroc_nonconf)
        bootstrapped_diffs.append(auroc_conf - auroc_nonconf)

    # Calculate statistics
    def get_stats(values):
        values = np.array(values)
        return (np.mean(values), np.percentile(values, [2.5, 97.5]))

    pos_mean, pos_ci = get_stats(bootstrapped_pos)
    neg_mean, neg_ci = get_stats(bootstrapped_neg)
    diff_mean, diff_ci = get_stats(bootstrapped_diffs)

    return pos_mean, pos_ci, neg_mean, neg_ci, diff_mean, diff_ci


def bootstrap_metrics_difference_threshold(label_index: int, confidence_level: float, confident_indices: np.ndarray,
                             non_confident_indices: np.ndarray, num_bootstraps: int,
                             all_outputs: np.ndarray, all_labels: np.ndarray,
                             bonferroni: bool = True, n_tests: int = 13):
    """
    Perform bootstrapping to estimate AUROC, AUPRC, and Sensitivity@95%Specificity metrics for both subsets and their differences.

    Args:
        bonferroni (bool): Whether to apply Bonferroni correction for multiple testing (default: True)
        n_tests (int): Number of simultaneous tests for Bonferroni correction (default: 13)

    Returns:
    - dict: Dictionary containing mean and CI for all metrics for both groups and their differences
    """
    bootstrapped_pos = {'auroc': [], 'auprc': [], 'sens_at_spec95': []}
    bootstrapped_neg = {'auroc': [], 'auprc': [], 'sens_at_spec95': []}
    bootstrapped_diffs = {'auroc': [], 'auprc': [], 'sens_at_spec95': []}

    for _ in tqdm(range(num_bootstraps), desc='Bootstrapping', unit='iter'):
        auroc_metric = torchmetrics.classification.MultilabelAUROC(num_labels=14, average='none')

        # Resample both subsets
        boot_outputs_conf, boot_labels_conf = resample(
            all_outputs[confident_indices],
            all_labels[confident_indices],
            stratify=all_labels[confident_indices]
        )
        boot_outputs_nonconf, boot_labels_nonconf = resample(
            all_outputs[non_confident_indices],
            all_labels[non_confident_indices],
            stratify=all_labels[non_confident_indices]
        )

        # Compute AUROC for confident group
        auroc_metric.update(boot_outputs_conf, boot_labels_conf)
        auroc_conf = auroc_metric.compute()[label_index]
        auroc_metric.reset()

        auroc_metric.update(boot_outputs_nonconf, boot_labels_nonconf)
        auroc_nonconf = auroc_metric.compute()[label_index]

        # Compute AUPRC for confident group
        y_true_conf = boot_labels_conf[:, label_index].cpu().numpy()
        y_pred_conf = boot_outputs_conf[:, label_index]
        auprc_conf = average_precision_score(y_true_conf, y_pred_conf)

        # Compute AUPRC for non-confident group
        y_true_nonconf = boot_labels_nonconf[:, label_index].cpu().numpy()
        y_pred_nonconf = boot_outputs_nonconf[:, label_index]
        auprc_nonconf = average_precision_score(y_true_nonconf, y_pred_nonconf)

        # Compute Sensitivity @ 95% Specificity for confident group
        fpr_conf, tpr_conf, _ = roc_curve(y_true_conf, y_pred_conf)
        target_fpr = 0.05  # 95% specificity = 5% FPR
        idx_conf = np.argmin(np.abs(fpr_conf - target_fpr))
        sens_conf = tpr_conf[idx_conf]

        # Compute Sensitivity @ 95% Specificity for non-confident group
        fpr_nonconf, tpr_nonconf, _ = roc_curve(y_true_nonconf, y_pred_nonconf)
        idx_nonconf = np.argmin(np.abs(fpr_nonconf - target_fpr))
        sens_nonconf = tpr_nonconf[idx_nonconf]

        # Store results
        bootstrapped_pos['auroc'].append(auroc_conf)
        bootstrapped_pos['auprc'].append(auprc_conf)
        bootstrapped_pos['sens_at_spec95'].append(sens_conf)

        bootstrapped_neg['auroc'].append(auroc_nonconf)
        bootstrapped_neg['auprc'].append(auprc_nonconf)
        bootstrapped_neg['sens_at_spec95'].append(sens_nonconf)

        bootstrapped_diffs['auroc'].append(auroc_conf - auroc_nonconf)
        bootstrapped_diffs['auprc'].append(auprc_conf - auprc_nonconf)
        bootstrapped_diffs['sens_at_spec95'].append(sens_conf - sens_nonconf)

    # Calculate statistics
    # Compute percentiles for confidence intervals (with optional Bonferroni correction)
    if bonferroni:
        corrected_alpha = 0.05 / n_tests
    else:
        corrected_alpha = 0.05

    lower_pct = (corrected_alpha / 2) * 100
    upper_pct = (1 - corrected_alpha / 2) * 100

    def get_stats(values):
        values = np.array(values)
        return {
            'mean': np.mean(values),
            'ci': tuple(np.percentile(values, [lower_pct, upper_pct]))
        }

    results = {
        'positive': {
            'auroc': get_stats(bootstrapped_pos['auroc']),
            'auprc': get_stats(bootstrapped_pos['auprc']),
            'sens_at_spec95': get_stats(bootstrapped_pos['sens_at_spec95'])
        },
        'negative': {
            'auroc': get_stats(bootstrapped_neg['auroc']),
            'auprc': get_stats(bootstrapped_neg['auprc']),
            'sens_at_spec95': get_stats(bootstrapped_neg['sens_at_spec95'])
        },
        'difference': {
            'auroc': get_stats(bootstrapped_diffs['auroc']),
            'auprc': get_stats(bootstrapped_diffs['auprc']),
            'sens_at_spec95': get_stats(bootstrapped_diffs['sens_at_spec95'])
        }
    }

    return results


def get_vision_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_dir="vision_predictions",
):
    """
    Run model inference to generate predictions and compute AUROC for a vision-based dataset.

    Parameters:
    - model (torch.nn.Module): Trained PyTorch model for generating predictions.
    - dataloader (torch.utils.data.DataLoader): Dataloader for loading validation/test images and labels.
    - output_dir (str): Directory to save predictions to.

    Returns:
    - torch.Tensor: The AUROC score of the model on the provided dataset.

    Saves:
    - '{output_dir}/full_predictions.pkl': File containing all model outputs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    auroc_metric = torchmetrics.classification.MultilabelAUROC(
        num_labels=14, average="none"
    )
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_outputs = []
    all_labels = []
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Evaluation",
    )
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():
            outputs = model(images)
        # Collect outputs and labels for later AUROC calculation
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())
        batch_bar.update()
        del images, labels
    batch_bar.close()
    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # Compute AUROC on the entire dataset at once
    auroc_metric.update(all_outputs, all_labels)
    auroc = auroc_metric.compute()
    print(f"Final AUROC: {auroc}")
    with open(f"{output_dir}/full_predictions.pkl", "wb") as f:
        pickle.dump(all_outputs, f)


def evaluate_multilabel_performance_with_bootstrapping(
    label_index: int,
    all_outputs: np.ndarray,
    all_labels: np.ndarray,
    num_labels: int = 14,
    num_bootstraps: int = 10000,
    confident_indices: dict = None,
    n_jobs: int = -1,
):
    """
    Evaluate multilabel classification performance with bootstrapping for confidence intervals.

    Parameters:
    - label_index (int): Index of the label being evaluated.
    - all_outputs (np.ndarray): Array of model predictions.
    - all_labels (np.ndarray): Array of true labels.
    - num_labels (int): Total number of labels in the multilabel setup.
    - num_bootstraps (int): Number of bootstrap samples.
    - confident_indices (dict): Dictionary mapping confidence levels to confident indices.
    - n_jobs (int): Number of parallel jobs for computation. (-1 for maximum available)

    Returns:
    - dict: Dictionary with confidence levels as keys and bootstrapped AUROC results as values.
    """
    # Convert to PyTorch Tensors for bootstrapping
    all_labels = torch.Tensor(all_labels).to(torch.int64)
    results = {}
    for confidence_level, indices in confident_indices.items():
        print(
            f"Evaluating label {label_index} at confidence level: {float(confidence_level)}"
        )
        # Get all indices and remove the confident ones to get non-confident indices
        all_indices = np.arange(len(all_outputs))
        non_confident_indices = np.setdiff1d(all_indices, indices)
        bootstrap_results = bootstrap_auroc(
            label_index,
            confidence_level,
            non_confident_indices,
            num_bootstraps,
            all_outputs,
            all_labels,
        )
        # Append the results
        results[confidence_level] = bootstrap_results

    return results


def evaluate_multilabel_performance_with_bootstrapping_diff(
    label_index: int,
    all_outputs: np.ndarray,
    all_labels: np.ndarray,
    num_labels: int = 14,
    num_bootstraps: int = 10000,
    confident_indices: dict = None,
):
    """
    Compute bootstrapped AUROC difference between confident and non-confident indices.

    Parameters:
    - label_index (int): Index of the label being evaluated.
    - all_outputs (np.ndarray): Array of model predictions.
    - all_labels (np.ndarray): Array of true labels.
    - num_labels (int): Total number of labels in the multilabel setup.
    - num_bootstraps (int): Number of bootstrap samples.
    - confident_indices (dict): Dictionary mapping confidence levels to confident indices.

    Returns:
    - dict: Dictionary with confidence levels as keys and bootstrapped AUROC difference results as values.
    """
    all_labels = torch.Tensor(all_labels).to(torch.int64)
    results = {}
    for confidence_level, indices in confident_indices.items():
        print(
            f"Evaluating label {label_index} at confidence level: {float(confidence_level)}"
        )
        # Get non-confident indices
        all_indices = np.arange(len(all_outputs))
        non_confident_indices = np.setdiff1d(all_indices, indices)
        # Compute AUROC difference with bootstrapping
        bootstrap_results = bootstrap_auroc_difference(
            label_index,
            indices,
            non_confident_indices,
            num_bootstraps,
            all_outputs,
            all_labels,
            num_labels,
        )
        results[confidence_level] = bootstrap_results
    return results


def bootstrap_auroc_difference(
    label_index: int,
    confident_indices: np.ndarray,
    non_confident_indices: np.ndarray,
    num_bootstraps: int,
    all_outputs: np.ndarray,
    all_labels: np.ndarray,
    num_labels: int,
):
    """
    Perform bootstrapping to estimate AUROC difference between confident and non-confident subsets.

    Parameters:
    - label_index (int): Index of the label being evaluated.
    - confident_indices (np.ndarray): Indices of confident samples.
    - non_confident_indices (np.ndarray): Indices of non-confident samples.
    - num_bootstraps (int): Number of bootstrap samples.
    - all_outputs (np.ndarray): Array of model outputs.
    - all_labels (np.ndarray): Array of true labels.
    - num_labels (int): Number of labels in the multilabel setup.

    Returns:
    - tuple: Mean AUROC difference and a confidence interval array [CI_low, CI_high].
    """
    bootstrapped_diffs = []
    auroc_metric = torchmetrics.classification.MultilabelAUROC(
        num_labels=num_labels, average="none"
    )
    for _ in range(num_bootstraps):
        # Resample confident and non-confident sets
        boot_conf_outputs, boot_conf_labels = resample(
            all_outputs[confident_indices],
            all_labels[confident_indices],
            stratify=all_labels[confident_indices],
        )
        boot_nonconf_outputs, boot_nonconf_labels = resample(
            all_outputs[non_confident_indices],
            all_labels[non_confident_indices],
            stratify=all_labels[non_confident_indices],
        )
        # Compute AUROC for confident subset
        auroc_metric.reset()
        auroc_metric.update(boot_conf_outputs, boot_conf_labels)
        auroc_confident = auroc_metric.compute()[label_index]
        # Compute AUROC for non-confident subset
        auroc_metric.reset()
        auroc_metric.update(boot_nonconf_outputs, boot_nonconf_labels)
        auroc_nonconfident = auroc_metric.compute()[label_index]
        # Store the AUROC difference
        bootstrapped_diffs.append(auroc_confident - auroc_nonconfident)
    # Compute mean and confidence intervals
    bootstrapped_diffs = torch.tensor(bootstrapped_diffs)
    mean_diff = torch.mean(bootstrapped_diffs)
    lower_percentile = np.percentile(bootstrapped_diffs, 2.5)
    upper_percentile = np.percentile(bootstrapped_diffs, 97.5)
    return mean_diff.item(), [lower_percentile, upper_percentile]


def save_metrics_to_csv(label: str, results: dict, evaluation_mode: str):
    """
    Save evaluation metrics for a specific label and confidence level to a CSV file.

    Parameters:
    - label (str): The label name.
    - results (dict): Dictionary with metrics results, containing confidence levels and AUROCs.
    - evaluation_mode (str): Mode of evaluation for metrics.

    Saves:
    - A CSV file with AUROC and confidence interval metrics.
    """
    output_file = f"vision_evaluation/evaluation_metrics_label_{label}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create CSV file with header if it doesn't exist
    if not os.path.exists(output_file):
        header = [
            "Label",
            "Evaluation_Mode",
            "Confidence_Level",
            "Mean_AUROC",
            "CI_Low",
            "CI_High",
        ]
        with open(output_file, "w") as file:
            file.write(",".join(header) + "\n")

    # Append the results to the CSV file
    for confidence_level, mean_auroc, auroc_ci in results:
        data_row = [
            label,
            evaluation_mode,
            confidence_level,
            mean_auroc,
            auroc_ci[0],
            auroc_ci[1],
        ]
        with open(output_file, "a") as file:
            file.write(",".join(map(str, data_row)) + "\n")


# Load high-confidence indices from .pkl files for each of the 14 labels
def load_high_confidence_indices(label: str, confident_dir="confident_predictions"):
    """
    Load high-confidence indices for a specific label.

    Parameters:
    - label (str): The label name to load confident indices for
    - confident_dir (str): Directory containing the confidence files

    Returns:
    - dict: Dictionary of high-confidence indices with confidence levels as keys
    """
    file_path = f"{confident_dir}/{label}.pkl"
    with open(file_path, "rb") as file:
        high_confidence_indices = pickle.load(file)
    return high_confidence_indices


######################
### Util Functions ###
######################

####################
### Util Classes ###
####################


class Mistral_Encoder(torch.utils.data.Dataset):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.float16)
        model_path = "mistralai/Mistral-7B-Instruct-v0.1"
        # model_path = "ml-experiments/data/models/Mistral-7B-Instruct-v0.1"
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          #local_files_only=True,
                                                          torch_dtype=torch.float16,
                                                          quantization_config=quantization_config,
                                                          #attn_implementation="flash_attention_2",
                                                          device_map="auto",
                                                          output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       #local_files_only=True
                                                       )

class CustomImageDataset(Dataset):
    def __init__(self, labels, img_path, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        image = Image.open(img_path).convert("RGB")
        labels = self.img_labels[idx].astype(np.int64)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)


####################
### Util Classes ###
####################
