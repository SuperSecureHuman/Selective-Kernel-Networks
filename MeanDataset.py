from torch.utils.data import DataLoader
import torchvision


test_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()
    ])

image_data = torchvision.datasets.ImageFolder('/home/venom/repo/SKNets/COVID-19_Radiography_Dataset/Balanced', transform=test_data_transform)

print(image_data.classes)

image_data_loader = DataLoader(
    image_data,
    batch_size=2048,
    shuffle=False,
    num_workers=0)

def mean_std(loader):
  images, lebels =  loader
  #print(images.shape)
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

for batch in image_data_loader:
  mean, std = mean_std(batch)
  print("mean and std: \n", mean, std)