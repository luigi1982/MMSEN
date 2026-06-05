import torch
import math
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


from src.data import load_data


ETA_0 = 1e-3
ETA_MIN = 1e-6

def lr_lambda(step: int, epochs: int) -> float:

    t = step + 1

    if t == 1:
        lr = ETA_0
    else:
        lr = ETA_0 + 0.5 * (ETA_0 - ETA_MIN) * (
            1 + math.cos(math.pi * (t - 1) / (epochs - 1))
        )

    return lr / ETA_0


def train_loop(model, experiment_name, train_transform, test_transform, epochs=10, custom_lr=False, lr=1e-3):

    ### test if google drive is connected
    save_path = "/content/drive/MyDrive/test.pt"
    torch.save({
        "this is a test": "this is a test"
    }, save_path)

    # get train and test data
    train_loader, test_loader, _ = load_data(train_transform, test_transform)
    # define the loss function
    loss_function = nn.BCEWithLogitsLoss()
    # get the device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # initialize the adam optimizer
    # initialize the learning rate schedule
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    if custom_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: lr_lambda(step, epochs))

    # lists to keep track of all metrics
    train_loss = []
    test_loss = []
    roc_auc = []
    precision = []
    sensitivity = []
    specificity = []
    f1_score = []
    balanced_accuracy = []
    mcc = []
    classes_per_epoch = []

    if 'cuda' in device:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for epoch in range(epochs):

        running_train_loss = 0.0
        running_test_loss = 0.0

        model.train()

        for batch in tqdm(train_loader, total=len(train_loader)):

            x = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
            y = batch[1].to(device, non_blocking=True).float()

            # zero the gradient
            optim.zero_grad(set_to_none=True)

            # forward pass
            # use bfloat to save computational power
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)

            # compute the loss
            loss = loss_function(out.squeeze(), y.squeeze())

            # backpropagate the error
            loss.backward()

            # update the weights
            optim.step()

            running_train_loss += loss.item()

            if custom_lr:
                scheduler.step()

            # report the train loss
            running_train_loss /= len(train_loader)
            print(f"[{epoch+1}/{epochs}] Train Loss: {running_train_loss:.3f}")

            all_targets = []
            all_scores = []
            classes = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

        model.eval()

        with torch.inference_mode():

            for batch in tqdm(test_loader, total=len(test_loader)):

                x = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
                y = batch[1].to(device, non_blocking=True).float()

                # forward pass
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)

                logits = logits.squeeze()
                y = y.squeeze()

                # compute the loss
                loss = loss_function(logits, y)

                running_test_loss += loss.item()

                # store predictions and labels for ROC-AUC
                all_scores.append(logits.detach().float().cpu())
                all_targets.append(y.detach().float().cpu())

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

            # compute roc - auc
            all_scores = torch.cat(all_scores).numpy()
            all_targets = torch.cat(all_targets).numpy()
            test_auc = roc_auc_score(all_targets, all_scores)

            # compute other metrics
            test_precision, test_sensitivity, test_specificity, test_f1_score, test_balanced_accuracy, test_mcc = compute_metrics(classes)
            roc_auc.append(test_auc)
            precision.append(test_precision)
            sensitivity.append(test_sensitivity)
            specificity.append(test_specificity)
            f1_score.append(test_f1_score)
            balanced_accuracy.append(test_balanced_accuracy)
            mcc.append(test_mcc)

            classes_per_epoch.append(classes)

            # append train and test loss
            running_test_loss /= len(test_loader)
            train_loss.append(running_train_loss)
            test_loss.append(running_test_loss)

            # report the test loss
            print(f"[{epoch+1}/{epochs}] Test Loss: {running_test_loss:.3f}")


    # save the model, only need to save the classifier
    save_path = f"/content/drive/MyDrive/{experiment_name}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'roc_auc': roc_auc,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score,
        'balanced_accuracy': balanced_accuracy,
        'mcc': mcc,
    }, save_path)

    return model, train_loss, test_loss, roc_auc, precision, sensitivity, specificity, f1_score, balanced_accuracy, mcc, classes_per_epoch

### same training loop, with regularization
### why did I implement an extra procedure?
### they should implement the same routine for lambda_ = 0

def train_loop_regularization(model, experiment_name, train_transform, test_transform, epochs=10, custom_lr=False, lr=1e-3, lambda_=0.0005, regularization_method='L1'):

    ### test if google drive is connected
    save_path = "/content/drive/MyDrive/test.pt"
    torch.save({
        "this is a test": "this is a test"
    }, save_path)

    # get train and test data
    train_loader, test_loader, _ = load_data(train_transform, test_transform)
    # define the loss function
    loss_function = nn.BCEWithLogitsLoss()
    # get the device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # initialize the adam optimizer
    # initialize the learning rate schedule
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    if custom_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: lr_lambda(step, epochs))

    # lists to keep track of all metrics
    objective = []
    regularization = []
    train_loss = []
    test_loss = []
    roc_auc = []
    precision = []
    sensitivity = []
    specificity = []
    f1_score = []
    balanced_accuracy = []
    mcc = []
    classes_per_epoch = []

    if 'cuda' in device:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for epoch in range(epochs):

        running_objective = 0.0
        running_regularization = 0.0
        running_train_loss = 0.0
        running_test_loss = 0.0

        model.train()

        for batch in tqdm(train_loader, total=len(train_loader)):

            x = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
            y = batch[1].to(device, non_blocking=True).float()

            # zero the gradient
            optim.zero_grad(set_to_none=True)

            # forward pass
            # use bfloat to save computational power
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)

            # compute the objective
            objective_ = loss_function(logits.squeeze(), y.squeeze())

            # compute regularization term
            regularization_ = 0.0

            if regularization_method == 'L1':
                regularization_ = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            elif regularization_method == 'L2':
                regularization_ = sum(torch.square(p).sum() for p in model.parameters() if p.requires_grad)

            # add objective and regularization to get loss
            loss = objective_ + lambda_ * regularization_

            # backpropagate the error
            loss.backward()

            # update the weights
            optim.step()

            running_objective += objective_.item()
            running_regularization += regularization_
            running_train_loss += loss.item()

            if custom_lr:
                scheduler.step()

            # report the train loss
            running_train_loss /= len(train_loader)
            running_objective /= len(train_loader)
            running_regularization *= lambda_/len(train_loader)
            print(f"[{epoch+1}/{epochs}] Train Loss: {running_train_loss:.3f} ; Objective: {running_objective:.3f}; Regularization: {running_regularization:.3f}")

            all_targets = []
            all_scores = []
            classes = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

        model.eval()

        with torch.inference_mode():

            for batch in tqdm(test_loader, total=len(test_loader)):

                x = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
                y = batch[1].to(device, non_blocking=True).float()

                # forward pass
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)

                logits = logits.squeeze()
                y = y.squeeze()

                # compute the loss
                loss = loss_function(logits.squeeze(), y.squeeze())

                running_test_loss += loss.item()

                # store predictions and labels for ROC-AUC
                all_scores.append(logits.detach().float().cpu())
                all_targets.append(y.detach().float().cpu())

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

            # compute roc - auc
            all_scores = torch.cat(all_scores).numpy()
            all_targets = torch.cat(all_targets).numpy()
            test_auc = roc_auc_score(all_targets, all_scores)

            # compute other metrics
            test_precision, test_sensitivity, test_specificity, test_f1_score, test_balanced_accuracy, test_mcc = compute_metrics(classes)
            roc_auc.append(test_auc)
            precision.append(test_precision)
            sensitivity.append(test_sensitivity)
            specificity.append(test_specificity)
            f1_score.append(test_f1_score)
            balanced_accuracy.append(test_balanced_accuracy)
            mcc.append(test_mcc)

            classes_per_epoch.append(classes)

            # append train and test loss
            running_test_loss /= len(test_loader)
            objective.append(running_objective)
            regularization.append(running_regularization)
            train_loss.append(running_train_loss)
            test_loss.append(running_test_loss)

            # report the test loss
            print(f"[{epoch+1}/{epochs}] Test Loss: {running_test_loss:.3f}")


    # save the model, only need to save the classifier
    save_path = f"/content/drive/MyDrive/{experiment_name}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'objective': objective,
        'regularization': regularization,
        'roc_auc': roc_auc,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score,
        'balanced_accuracy': balanced_accuracy,
        'mcc': mcc,
    }, save_path)

    return model, train_loss, test_loss, objective, regularization, roc_auc, precision, sensitivity, specificity, f1_score, balanced_accuracy, mcc, classes_per_epoch