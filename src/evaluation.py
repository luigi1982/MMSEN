import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data import load_data

def comp_precision(classes):
  
    '''
        precision:
        returns the percentage of samples that have been accurately classified as positive by the model
    '''

    return classes['TP'] / (classes['TP'] + classes['FP'] + 1e-5)


def comp_sensitivity(classes):
  
    '''
        sensitivity:
        returns the percentage of positive samples that have been detected by the model
    '''

    return classes['TP'] / (classes['TP'] + classes['FN'] + 1e-5)


def comp_specificity(classes):

    '''
        specifity:
        returns the percentage of negative samples that have been correctly detected by the model
    '''

    return classes['TN'] / (classes['TN'] + classes['FP'] + 1e-5)


def comp_f1_score(classes):

    '''
        F1-Score:
        return the harmonic mean of precision and sensitivity
    '''

    prec = comp_precision(classes)
    sens = comp_sensitivity(classes)
    return 2 * (prec * sens) / (prec + sens + 1e-5)


def comp_balanced_accuracy(classes):

    '''
        Balanced Accuracy:
        returns the mean of sensitivity and specifity
    '''

    sens = comp_sensitivity(classes)
    spec = comp_specificity(classes)

    return (sens + spec) / 2


def comp_mcc(classes):

    '''
        Matthews-Correlation-Coefficient
    '''

    return (classes['TP'] * classes['TN'] - classes['FP'] * classes['FN'])/((classes['TP'] + classes['FP']) * (classes['TP'] + classes['FN']) * (classes['TN'] + classes['FP']) * (classes['TN'] + classes['FN']))**0.5


def compute_metrics(classes):
    
    '''
        Return all of the metrics
    '''

    return comp_precision(classes), comp_sensitivity(classes), comp_specificity(classes), comp_f1_score(classes), comp_balanced_accuracy(classes), comp_mcc(classes)


### plot train and test error as well as all of the above metrics inside a 4x2 grid

def plot_metrics(train_loss, test_loss, roc_auc, precision, sensitivity, specificity, f1_score, balanced_accuracy, mcc):

    fig, axs = plt.subplots(2, 4)

    # train test loss
    axs[0, 0].plot(train_loss, 'o-')
    axs[0, 0].plot(test_loss, 'o-')
    axs[0, 0].legend(['train', 'test'])
    axs[0, 0,].set_title('Train and Test Loss')

    #roc auc
    axs[0, 1].plot(roc_auc, 'o-')
    axs[0, 1].set_title('ROC- AUC')

    #precision
    axs[0, 2].plot(precision, 'o-')
    axs[0, 2].set_title('Precision')

    #sensitivity
    axs[0, 3].plot(sensitivity, 'o-')
    axs[0, 3].set_title('Sensitivity')

    #specificity
    axs[1, 0].plot(specificity, 'o-')
    axs[1, 0].set_title('Specifity')

    #f1_score
    axs[1, 1].plot(f1_score, 'o-')
    axs[1, 1].set_title('F1 - Score')

    #balanced_accuracy
    axs[1, 2].plot(balanced_accuracy, 'o-')
    axs[1, 2].set_title('Balanced Accuracy')

    #mcc
    axs[1, 3].plot(mcc, 'o-')
    axs[1, 3].set_title('MCC')

    fig.tight_layout()
    plt.plot()


### create a confusion matrix from FP, TP, FN, TN
def confusion_matrix(classes):

    cm = np.array([[classes['FP'], classes['TP']], [classes['TN'], classes['FN']]])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    # Axis ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Pred 1', 'Pred 0'])
    ax.set_xticklabels(['True 0', 'True 1'])

    # Annotate each cell
    labels = [['FP', 'TP'],
                ['TN', 'FN']]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{int(cm[i, j])}",
                    ha='center', va='center')

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")

    plt.colorbar(im)
    plt.show()


### compare multiple models and experiments
def compare_exps(exps, labels):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 5), constrained_layout=True)

    for ax in axes:
        ax.set_box_aspect(1)

    for exp, label in zip(exps, labels):

        mmsen_stats = torch.load('/content/drive/MyDrive/'+exp+'.pth', weights_only=False, map_location=torch.device(device))

        # test loss
        test_loss_mmsen = mmsen_stats['test_loss']
        # roc - auc
        roc_auc_mmsen = mmsen_stats['roc_auc']
        # precision
        precision_mmsen = mmsen_stats['precision']
        # sensitivity
        sensitivity_mmsen = mmsen_stats['sensitivity']
        # specifity
        specifity_mmsen = mmsen_stats['specificity']

        axes[0].plot(test_loss_mmsen, 'o-', label=label)
        axes[1].plot(roc_auc_mmsen, 'o-', label=label)
        axes[2].plot(precision_mmsen, 'o-', label=label)
        axes[3].plot(sensitivity_mmsen, 'o-', label=label)
        axes[4].plot(specifity_mmsen, 'o-', label=label)

        axes[0].set_title('Test Loss')
        axes[1].set_title('ROC-AUC')
        axes[2].set_title('Precision')
        axes[3].set_title('Sensitivity')
        axes[4].set_title('Specifity')

    # Hide the unused 6th subplot
    axes[-1].axis("off")

    # Put one shared legend in the empty bottom-right space
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center")

    plt.tight_layout()
    plt.legend()
    plt.show()


### visualize images, prediction and ground truth for a batch of images
def vis_results(model, train_transform, test_transform, mean, std):

    _, test_loader, _ = load_data(train_transform, test_transform)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()

    test_batch = next(iter(test_loader))

    x = test_batch[0].to(device)
    y = test_batch[1]

    with torch.inference_mode():
        prediction = model(x).cpu()

    prediction = torch.round(torch.sigmoid(prediction))

    invTrans = transforms.Compose([
        transforms.Normalize(mean = torch.zeros(3), std = torch.ones(3)/std),
        transforms.Normalize(mean = -mean, std = torch.ones(3)),
    ])

    x = invTrans(x)

    fig, axs = plt.subplots(2, 5)

    for i in range(2):
        for j in range(5):

            axs[i, j].imshow(x[5*i + j].cpu().permute(1, 2, 0))

            # Remove axis ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

            # Optional: remove the whole axis frame
            axs[i, j].axis('off')

            # Add title
            title = 'GT: ' + ('P 'if y[5*i + j].item() else 'N') + ' Pred: ' + ('P 'if prediction[5*i + j].item() else 'N')
            axs[i, j].set_title(title)

    fig.tight_layout()
    plt.show()


def eval(model, test_transform):

    _, _, val_loader = load_data(None, test_transform)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    model.eval()

    with torch.inference_mode():

        for batch in tqdm(val_loader, total=len(val_loader)):

            x = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
            y = batch[1].to(device, non_blocking=True).float()

            # forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)

            logits = logits.squeeze()
            y = y.squeeze()

            # get the label from the prediction
            label_pred = torch.round(torch.sigmoid(logits))

            # get the TPs, FPs, TNs, FNs
            tp = torch.sum(label_pred[y == 1])
            fp = torch.sum(label_pred[y == 0])
            fn = torch.sum(y) - tp

            classes['TP'] += tp.item()
            classes['FP'] += fp.item()
            classes['FN'] += fn.item()
            classes['TN'] += y.numel() - tp.item() - fp.item() - fn.item()

    return classes