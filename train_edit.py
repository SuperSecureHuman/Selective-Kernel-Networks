import argparse

import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import wandb

import sknet
from utils import train, test, EarlyStopping


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default="SKNet26")
parser.add_argument('--wandb', type=bool, default=False)
parser.add_argument('--data', type=str, default="./Images")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--image_size', type=int, default=224)

args = parser.parse_args()

##############################################
# Hyperparameters
##############################################
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
MODEL = args.model
DATA = args.data
SEED = args.seed
IMAGE_SIZE = args.image_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 0
RUN_NAME = MODEL + "_lr_" +  \
    str(LEARNING_RATE) + "_weight_decay_" + str(WEIGHT_DECAY) + "_epoch_" + \
    str(EPOCHS) + "_img_size_" + str(IMAGE_SIZE) + "_seed_" + str(SEED) + "_Normalised_HPC"

print("RUNNING ON DEVICE: ", DEVICE)
print(DATA)

##############################################
# some flags
##############################################
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_fp16_reduced_precision_reduction = True


##############################################
# wandb
##############################################
if args.wandb == True:
    wandb.init(
        project="Selective Kernel Networks",
        name=RUN_NAME,

        sync_tensorboard=True,

        save_code=True,

        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "architecture": MODEL,
            "dataset": "COVID-19 Radiography",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "seed": SEED,
        }
    )
else:
    pass

model = getattr(sknet, MODEL)(nums_class=4).to(DEVICE)


##############################################
# Dataset
##############################################
train_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(degrees=30),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


test_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = torchvision.datasets.ImageFolder(root=DATA, transform=train_data_transform)
train_ds, _, _ = random_split(dataset, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(SEED))

dataset = torchvision.datasets.ImageFolder(root=DATA, transform=test_data_transform)
_, val_ds, test_ds = random_split(dataset, [0.7, 0.2,0.1], generator=torch.Generator().manual_seed(SEED))


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True , num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False , num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False , num_workers=4, pin_memory=True)


##############################################
# Tensorboard
##############################################
tb_name = "tensorboard/" + RUN_NAME
writer = torch.utils.tensorboard.SummaryWriter(tb_name, comment=RUN_NAME)
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
writer.add_image("images", grid)
images = images.to(DEVICE)
writer.add_graph(model, images)
writer.close()


##############################################
# Training
##############################################
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

early_stopping = EarlyStopping(
    patience=5, verbose=True, path=RUN_NAME+'_best.pt')

total_step = len(train_loader)
highest_accuracy = 0
temp_high = 0
for epoch in range(EPOCHS):
    print("Epoch: ", epoch+1, " / " + str(EPOCHS))
    train_loss, train_correct, train_total = train(
        model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_correct, val_total, recall, precision = test(
        model, val_loader, criterion, DEVICE)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}%, Test Loss: {:.4f}, Test Acc: {:.4f}%'.format(
        epoch+1, EPOCHS, train_loss, 100 * train_correct / train_total, val_loss, 100 * val_correct / val_total))

    # print pres and recall and f1 score
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1 Score: ", (2*recall*precision) / (recall+precision))

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        temp_high = 100 * val_correct / val_total
        if temp_high > highest_accuracy:
            highest_accuracy = temp_high
        early_stopping.path = './best/' + RUN_NAME+'_best_' + str(temp_high) + '.pt'
        print("Early stopping")
        early_stopping.early_stop = False
        early_stopping.counter = 0
        early_stopping.best_score = None

    writer.add_scalar('training_loss', train_loss, epoch)
    writer.add_scalar('training_accuracy', 100 *
                      train_correct / train_total, epoch)

    writer.add_scalar('Val_Loss', val_loss, epoch)
    writer.add_scalar('Val_Acc', 100 *
                      val_correct / val_total, epoch)

    writer.add_scalar('val_recall', recall, epoch)
    writer.add_scalar('val_precision', precision, epoch)
    writer.add_scalar('val_f1_score', (2*recall*precision) /
                      (recall+precision), epoch)

    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, epoch)
        writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    print("\n")


model_path = "models/" + RUN_NAME + ".pt"
torch.save(model.state_dict(), model_path)

print("-----------Test Set Results-----------")

model = getattr(sknet, MODEL)(nums_class=4).to(DEVICE)

model.load_state_dict(torch.load('./best/'  + RUN_NAME+'_best_' + str(highest_accuracy) + '.pt'))

test_loss, test_correct, test_total, test_recall, test_pres = test(
    model, test_loader, criterion, DEVICE)


writer.add_scalar('Test_Loss', test_loss, EPOCHS)
writer.add_scalar('TestAcc', 100 *
                  test_correct / test_total, EPOCHS)
writer.add_scalar('val_recall', test_recall, EPOCHS)
writer.add_scalar('val_precision', test_pres, EPOCHS)
writer.add_scalar('val_f1_score', (2*test_recall*test_pres) /
                  (test_recall+test_pres), EPOCHS)

writer.flush()
writer.close()

if args.wandb == True:
    #wandb.save(model_path)
    wandb.finish()
