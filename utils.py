#https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

from tqdm import tqdm
import torch
import numpy as np

from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall

# Gradient scaler for amp (Mixed Precision)
scaler = torch.cuda.amp.GradScaler()


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_sum = 0
    len_loader = len(train_loader)
    p_tqdm = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(p_tqdm):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        #    output = model(data)
        #    loss = criterion(output, target)
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        # update loss in each batch in progress bar
        train_loss = loss.item()
        loss_sum += loss.item()
        p_tqdm.set_postfix_str(
            f"Train Loss: {train_loss:.3f} | Train Acc: {100.*correct/total:.3f}")

    return loss_sum/len_loader, correct, total


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_sum = 0
    len_loader = len(test_loader)

    recall = MulticlassRecall(num_classes=4).to(device)
    precision = MulticlassPrecision(num_classes=4).to(device)
    prec, rec = [], []

    with torch.no_grad():
        p_tqdm = tqdm(test_loader)
        for batch_idx, (data, target) in enumerate(p_tqdm):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss = loss.item()
            loss_sum += loss.item()
            _, predicted = output.max(1)
            rec.append(recall(predicted, target))
            prec.append(precision(predicted, target))
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            p_tqdm.set_postfix_str(
                f"Test Loss: {test_loss:.3f} | Test Acc: {100.*correct/total:.3f}")

    return loss_sum/len_loader, correct, total, sum(rec)/len(test_loader), sum(prec)/len(test_loader)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
