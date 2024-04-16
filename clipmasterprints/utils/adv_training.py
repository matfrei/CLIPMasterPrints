import torch
from torch.nn.functional import cross_entropy
import torchvision

@torch.no_grad()
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def split_adv_dataset(adv_dataset_path, train_size, val_size, test_size, preprocessing=None):
    train_set_size_half = int(train_size/2)
    val_set_size_half = int(val_size/2)
    test_set_size_half = int(test_size/2)

    val_set_size_half_cum = train_set_size_half + val_set_size_half
    test_set_size_half_cum = train_set_size_half + val_set_size_half + test_set_size_half

    train_val_set_adv = torchvision.datasets.ImageFolder(root=adv_dataset_path, transform=preprocessing)
    print(train_val_set_adv.class_to_idx)

    targets = torch.tensor(train_val_set_adv.targets)

    train_and_val_tainted_idxs = (targets == 1).nonzero().view(-1)
    train_and_val_tainted_idxs = train_and_val_tainted_idxs[
                                 :(train_set_size_half + val_set_size_half + test_set_size_half)].tolist()

    train_and_val_pure_idxs = (targets == 0).nonzero().view(-1)
    train_and_val_pure_idxs = train_and_val_pure_idxs[
                              :train_set_size_half + val_set_size_half + test_set_size_half].tolist()

    train_idx = train_and_val_tainted_idxs[:train_set_size_half] + train_and_val_pure_idxs[:train_set_size_half]
    val_idx = train_and_val_tainted_idxs[train_set_size_half:val_set_size_half_cum] + train_and_val_pure_idxs[
                                                                                      train_set_size_half:val_set_size_half_cum]
    test_idx = train_and_val_tainted_idxs[val_set_size_half_cum:test_set_size_half_cum] + train_and_val_pure_idxs[
                                                                                          val_set_size_half_cum:test_set_size_half_cum]

    train_set = torchvision.datasets.ImageFolder(root=adv_dataset_path, transform=preprocessing)
    train_set.targets = [train_set.targets[idx] for idx in train_idx]
    train_set.samples = [train_set.samples[idx] for idx in train_idx]

    val_set = torchvision.datasets.ImageFolder(root=adv_dataset_path, transform=preprocessing)
    val_set.targets = [val_set.targets[idx] for idx in val_idx]
    val_set.samples = [val_set.samples[idx] for idx in val_idx]

    test_set = torchvision.datasets.ImageFolder(root=adv_dataset_path, transform=preprocessing)
    test_set.targets = [test_set.targets[idx] for idx in test_idx]
    test_set.samples = [test_set.samples[idx] for idx in test_idx]

    return train_set, val_set, test_set


def train_adv(model, learning_rate, num_epochs, dataloaders, params, save_callback, loss=cross_entropy):
    train_loss_mean = params[0].data.new_full((1, 1), fill_value=0.)
    [train_dataloader, val_dataloader] = dataloaders
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    min_test_loss = 20.
    for epoch in range(1, num_epochs+1):
        for i_batch, batch in enumerate(train_dataloader):
            images = batch[0].cuda()
            labels = batch[1].cuda()
            output = model(images)
            train_loss = loss(output,labels)
            optimizer.zero_grad()
            train_loss_mean = 0.6 * train_loss.detach() + 0.4 * train_loss_mean
            train_loss.backward()
            optimizer.step()

        print('Epoch {:02.2f}: normalized train loss (smoothed): {:02.2f} , lr: {}'.format(epoch, train_loss_mean.item(), optimizer.param_groups[0]['lr']))

        del images,labels, output
        test_loss,accuracy = test_adv(model, val_dataloader,loss)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            save_callback(model,epoch)

        print('======================================================================')
        print('Epoch {:02.2f}: validation loss: {:02.2f}, validation acc: {:02.4f}'.format(epoch,test_loss, accuracy))
        print('======================================================================')
        model = model.train()


def test_adv(model,test_dataloader,loss=cross_entropy):
    model = model.eval()
    test_loss = 0.

    num_correct = 0
    for i_batch, test_batch in enumerate(test_dataloader):
        test_images = test_batch[0].cuda()
        test_labels = test_batch[1].cuda()

        with torch.no_grad():
            predictions = model(test_images)
            test_loss += loss(predictions, test_labels).item()
            num_correct += get_num_correct(predictions, test_labels)

    test_loss = test_loss / len(test_dataloader)
    accuracy = num_correct / len(test_dataloader.dataset)
    return test_loss, accuracy