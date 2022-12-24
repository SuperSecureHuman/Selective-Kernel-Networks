from tqdm import tqdm
import torch

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
