import torch,torchvision
from torch.nn.functional import cross_entropy
from clipmasterprints import test_adv, split_adv_dataset

# params
batch_size = 152
device = torch.device("cuda:0")

# train and test transforms
input_size = 224

pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

preprocessing = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(input_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,max_size=None, antialias=True),
    torchvision.transforms.CenterCrop(size=(input_size,input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
])

train_path_adv = "~/data/adv_full_proc/train/"
train_set, val_set, test_set = split_adv_dataset(train_path_adv,60000,10000,10000,preprocessing=preprocessing)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

vgg19 = torchvision.models.vgg19_bn()

#replace output layer
output_layer = torch.nn.Linear(in_features = 4096, out_features = 2)
vgg19.classifier[6] = output_layer

vgg19.load_state_dict(torch.load('results/cmp_detector/vgg19_detect_cmp_1ep.pt'))
vgg19.eval().to(device)

train_loss,train_accuracy = test_adv(vgg19,train_dataloader,cross_entropy)
print(f"train accuracy: {train_accuracy:.4f}")

val_loss,val_accuracy = test_adv(vgg19,val_dataloader,cross_entropy)
print(f"val accuracy: {val_accuracy:.4f}")

test_loss,test_accuracy = test_adv(vgg19,test_dataloader,cross_entropy)
print(f"test accuracy: {test_accuracy:.4f}")