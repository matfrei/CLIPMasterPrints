import torch,torchvision
from clipmasterprints import train_adv, test_adv, split_adv_dataset

# hyperparams
batch_size = 152
learning_rate = 0.001
num_epochs = 5
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

train_set,val_set,test_set = split_adv_dataset(train_path_adv, 60000, 10000, 10000, preprocessing=preprocessing)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

vgg19 = torchvision.models.vgg19_bn(weights="IMAGENET1K_V1")

# replace output layer
output_layer = torch.nn.Linear(in_features = 4096, out_features = 2)
vgg19.classifier[6] = output_layer

params = list(vgg19.parameters())
train_loss_mean = params[0].data.new_full((1, 1), fill_value=0.)
vgg19 = vgg19.to(device).train()
train_adv(vgg19,learning_rate,num_epochs,[train_dataloader,val_dataloader],params,save_callback=lambda model,epoch: torch.save(model.state_dict(),'results-reproduced/cmp_detectors/vgg19_detect_cmp.pt'))
test_loss,test_accuracy = test_adv(vgg19,test_dataloader)
print(f"test accuracy: {test_accuracy:.4f}")