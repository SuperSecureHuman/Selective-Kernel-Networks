import argparse

import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import wandb

import sknet
from utils import train, test


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
    str(EPOCHS) + "_img_size_" + str(IMAGE_SIZE) + "_seed_" + str(SEED)

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
            "architecture": "SKNet26",
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
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder(
    root=DATA, transform=transform)
train_len = int(len(dataset) * 0.7)
test_len = len(dataset) - train_len
train_ds, test_ds = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)# , num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True) # , num_workers=4, pin_memory=True)


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

total_step = len(train_loader)
highest_accuracy = 0
for epoch in range(EPOCHS):
    print("Epoch: ", epoch+1, " / " + str(EPOCHS))
    train_loss, train_correct, train_total = train(
        model, train_loader, optimizer, criterion, DEVICE)
    test_loss, test_correct, test_total, recall, precision = test(
        model, test_loader, criterion, DEVICE)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}%, Test Loss: {:.4f}, Test Acc: {:.4f}%'.format(
        epoch+1, EPOCHS, train_loss, 100 * train_correct / train_total, test_loss, 100 * test_correct / test_total))

    writer.add_scalar('training_loss', train_loss, epoch)
    writer.add_scalar('training_accuracy', 100 *
                      train_correct / train_total, epoch)

    writer.add_scalar('testing_loss', test_loss, epoch)
    writer.add_scalar('testing_accuracy', 100 *
                      test_correct / test_total, epoch)

    writer.add_scalar('recall', recall, epoch)
    writer.add_scalar('precision', precision, epoch)
    writer.add_scalar('f1_score', (2*recall*precision) /
                      (recall+precision), epoch)

    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, epoch)
        writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    print("\n")

    test_acc = 100 * test_correct / test_total

    if test_acc > highest_accuracy:
        highest_accuracy = test_acc
        torch.save(model.state_dict(), 'models/' + RUN_NAME +  'best_test.pth')

writer.flush()
writer.close()

model_path = "models/" + RUN_NAME + ".pth"

torch.save(model.state_dict(), model_path)

if args.wandb == True:
    #wandb.save(model_path)
    wandb.finish()
