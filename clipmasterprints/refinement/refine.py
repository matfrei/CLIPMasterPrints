# adapted from refining clip code from github user vinson2233 on https://github.com/openai/CLIP/issues/83
# mixed in approaches from open clip training script https://github.com/mlfoundations/open_clip
# thanks!
import torch,torchvision
import clip
from datetime import datetime

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def print_and_log(message,log_fn):
    if log_fn is not None:
        log_fn(message)
    print(message)

def refine_clip_mp(model, learning_rate, num_epochs, dataloaders, device,tb_writer,save_fn=lambda model,epoch,loss: torch.save(model.state_dict(),f'./model_epoch{epoch}'),log_fn=None):
    params = list(model.parameters())
    train_loss_mean = None
    [train_dataloader, test_dataloader] = dataloaders
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    # convert to fp32 for mixed-precision training
    # forward path will be fp16 using torch.autocast
    convert_models_to_fp32(model)

    for epoch in range(1, num_epochs + 1):
        for (i_batch, (images,captions)) in enumerate(train_dataloader):
            optimizer.zero_grad()
            #TODO: tokenize in dataloader for speedup!
            tokens = clip.tokenize(captions).to(device)
            images = images.to(device)
            with torch.autocast(device_type=device.type):
                logits_per_image, logits_per_text = model(images, tokens)
                ground_truth = torch.arange(images.shape[0], dtype=torch.long).to(device)
                loss_norm_image = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
                loss_norm_text = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
                train_loss = (loss_norm_image + loss_norm_text) / 2

            train_loss.backward()
            optimizer.step()
            train_loss_mean = train_loss.detach() if train_loss_mean is None else 0.1 * train_loss.detach() + 0.9 * train_loss_mean
            tb_writer.add_scalar("train loss smooth", train_loss_mean, epoch-1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss", train_loss, epoch-1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss image", loss_norm_image, epoch-1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss text", loss_norm_text, epoch-1 + (float(i_batch) / len(train_dataloader)))

            if i_batch % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                now = datetime.now()
                current_time = now.strftime("%d.%m.%Y, %H:%M:%S")
                print_and_log(
                    f'{current_time}, Epoch {epoch-1 + (float(i_batch) / len(train_dataloader)):02.2f}: train loss smoothed: {train_loss_mean:02.2f}, train loss batch: {train_loss:02.2f}, image loss: {loss_norm_image:02.2f}, text loss: {loss_norm_text:02.2f}, lr: {current_lr}',
                    log_fn)

        current_lr = optimizer.param_groups[0]['lr']
        now = datetime.now()
        current_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        print_and_log(f'{current_time}, Epoch {epoch -1 + (float(i_batch) / len(train_dataloader)): 02.2f}: train loss smoothed: {train_loss_mean:02.2f}, train loss batch: {train_loss:02.2f}, image loss: {loss_norm_image:02.2f},text loss: {loss_norm_text:02.2f}, lr: {current_lr}',log_fn)

        # now every 5 epochs, validation loss may be interesting
        if epoch % 1 == 0:
            model = model.eval()
            test_loss = 0.
            clip.model.convert_weights(model)
            for (i_batch, (images, captions)) in enumerate(test_dataloader):
                with torch.no_grad():
                    #TODO: tokenize in dataloader for speed!
                    tokens = clip.tokenize(captions).to(device)
                    images = images.to(device)
                    logits_per_image, logits_per_text = model(images, tokens)
                    ground_truth = torch.arange(images.shape[0], dtype=torch.long).to(device)
                    loss_norm_image = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
                    loss_norm_text = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
                    test_loss += (loss_norm_image + loss_norm_text) / 2

            test_loss = test_loss / len(test_dataloader)
            tb_writer.add_scalar("validation loss", test_loss, epoch)

            save_fn(model,epoch,test_loss)

            print_and_log('======================================================================', log_fn)
            print_and_log('Epoch {:02.2f}: test loss: {:02.2f}'.format(epoch, test_loss), log_fn)
            print_and_log('======================================================================', log_fn)

            convert_models_to_fp32(model)
            model = model.train()
        tb_writer.flush()



def refine_clip_wrong_token_random(model, learning_rate, num_epochs, dataloaders, device, tb_writer, wrong_image,
                       latent_decoder,save_fn=lambda model, epoch, loss: torch.save(model.state_dict(), f'./model_epoch{epoch}'),
                       log_fn=None):

    params = list(model.parameters())
    train_loss_mean = None
    [train_dataloader, test_dataloader] = dataloaders
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    wrong_image = wrong_image.to(device)
    clip_input_res = model.visual.input_resolution
    input_mean = train_dataloader.dataset.transform.transforms[-1].mean
    input_std = train_dataloader.dataset.transform.transforms[-1].std

    tensor_preprocessing = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(clip_input_res,clip_input_res), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    torchvision.transforms.CenterCrop(size=(clip_input_res,clip_input_res)),
    torchvision.transforms.Normalize(mean=input_mean, std=input_std)])

    # convert to fp32 for mixed-precision training
    # forward path will be fp16 using torch.autocast
    convert_models_to_fp32(model)

    for epoch in range(1, num_epochs + 1):
        for (i_batch, (images, captions)) in enumerate(train_dataloader):
            optimizer.zero_grad()
            captions.extend(['<off-manifold>','<off-manifold>'])
            # TODO: tokenize in dataloader for speedup!
            tokens = clip.tokenize(captions).to(device)
            images = images.to(device)
            with torch.no_grad():
                noise_image = tensor_preprocessing(latent_decoder.to_image(torch.randn((1,3,224//8,224//8),device=device)))
            #TODO: next step: sample the cma object rather than just using the static image
            images = torch.cat([images,wrong_image,noise_image],dim=0)
            with torch.autocast(device_type=device.type):
                logits_per_image, logits_per_text = model(images, tokens)
                ground_truth = torch.arange(images.shape[0], dtype=torch.long).to(device)
                loss_norm_image = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
                loss_norm_text = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
                train_loss = (loss_norm_image + loss_norm_text) / 2

            train_loss.backward()
            optimizer.step()
            train_loss_mean = train_loss.detach() if train_loss_mean is None else 0.1 * train_loss.detach() + 0.9 * train_loss_mean
            tb_writer.add_scalar("train loss smooth", train_loss_mean,
                                 epoch - 1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss", train_loss, epoch - 1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss image", loss_norm_image,
                                 epoch - 1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss text", loss_norm_text,
                                 epoch - 1 + (float(i_batch) / len(train_dataloader)))

            if i_batch % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                now = datetime.now()
                current_time = now.strftime("%d.%m.%Y, %H:%M:%S")
                print_and_log(
                    f'{current_time}, Epoch {epoch - 1 + (float(i_batch) / len(train_dataloader)):02.2f}: train loss smoothed: {train_loss_mean:02.2f}, train loss batch: {train_loss:02.2f}, image loss: {loss_norm_image:02.2f}, text loss: {loss_norm_text:02.2f}, lr: {current_lr}',
                    log_fn)

        current_lr = optimizer.param_groups[0]['lr']
        now = datetime.now()
        current_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        print_and_log(
            f'{current_time}, Epoch {epoch - 1 + (float(i_batch) / len(train_dataloader)): 02.2f}: train loss smoothed: {train_loss_mean:02.2f}, train loss batch: {train_loss:02.2f}, image loss: {loss_norm_image:02.2f},text loss: {loss_norm_text:02.2f}, lr: {current_lr}',
            log_fn)

        # now every 5 epochs, validation loss may be interesting
        if epoch % 1 == 0:
            model = model.eval()
            test_loss = 0.
            clip.model.convert_weights(model)
            for (i_batch, (images, captions)) in enumerate(test_dataloader):
                with torch.no_grad():
                    # TODO: tokenize in dataloader for speed!
                    tokens = clip.tokenize(captions).to(device)
                    images = images.to(device)
                    logits_per_image, logits_per_text = model(images, tokens)
                    ground_truth = torch.arange(images.shape[0], dtype=torch.long).to(device)
                    loss_norm_image = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
                    loss_norm_text = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
                    test_loss += (loss_norm_image + loss_norm_text) / 2

            test_loss = test_loss / len(test_dataloader)
            tb_writer.add_scalar("validation loss", test_loss, epoch)

            save_fn(model, epoch, test_loss)

            print_and_log('======================================================================', log_fn)
            print_and_log('Epoch {:02.2f}: test loss: {:02.2f}'.format(epoch, test_loss), log_fn)
            print_and_log('======================================================================', log_fn)

            convert_models_to_fp32(model)
            model = model.train()
        tb_writer.flush()

def refine_clip_wrong_token_loop(model, learning_rate, num_epochs, dataloaders, device, tb_writer, cma_optimizer,
                       latent_decoder,save_fn=lambda model, epoch, loss: torch.save(model.state_dict(), f'./model_epoch{epoch}'),
                       log_fn=None):

    params = list(model.parameters())
    train_loss_mean = None
    [train_dataloader, test_dataloader] = dataloaders
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)


    clip_input_res = model.visual.input_resolution
    input_mean = train_dataloader.dataset.transform.transforms[-1].mean
    input_std = train_dataloader.dataset.transform.transforms[-1].std

    tensor_preprocessing = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(clip_input_res,clip_input_res), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    torchvision.transforms.CenterCrop(size=(clip_input_res,clip_input_res)),
    torchvision.transforms.Normalize(mean=input_mean, std=input_std)])

    # convert to fp32 for mixed-precision training
    # forward path will be fp16 using torch.autocast
    convert_models_to_fp32(model)

    for epoch in range(1, num_epochs + 1):
        for (i_batch, (images, captions)) in enumerate(train_dataloader):
            optimizer.zero_grad()
            captions.extend(['<off-manifold>','<off-manifold>'])
            # TODO: tokenize in dataloader for speedup!
            tokens = clip.tokenize(captions).to(device)
            images = images.to(device)

            with torch.no_grad():
                # run cma optimizer to adapt solution to updated model
                clip.model.convert_weights(model)
                with torch.no_grad():
                    cma_optimizer.step()
                convert_models_to_fp32(model)
                # convert ES solution to tensor in decoder latent space
                updated_solution_latent = torch.from_numpy(cma_optimizer.es.result.xfavorite.reshape(1,4,28,28).astype('float32')).to(device)
                decoder_inputs = torch.cat([updated_solution_latent,torch.randn((1,4,28,28),device=device)],dim=0)
                # generate off-manifold images using SD decoder
                off_manifold_images = tensor_preprocessing(latent_decoder.to_image(decoder_inputs))
                images = torch.cat([images,off_manifold_images],dim=0)

            with torch.autocast(device_type=device.type):
                logits_per_image, logits_per_text = model(images, tokens)
                ground_truth = torch.arange(images.shape[0], dtype=torch.long).to(device)
                loss_norm_image = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
                loss_norm_text = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
                train_loss = (loss_norm_image + loss_norm_text) / 2

            train_loss.backward()
            optimizer.step()
            train_loss_mean = train_loss.detach() if train_loss_mean is None else 0.1 * train_loss.detach() + 0.9 * train_loss_mean
            tb_writer.add_scalar("train loss smooth", train_loss_mean,
                                 epoch - 1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss", train_loss, epoch - 1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss image", loss_norm_image,
                                 epoch - 1 + (float(i_batch) / len(train_dataloader)))
            tb_writer.add_scalar("train loss text", loss_norm_text,
                                 epoch - 1 + (float(i_batch) / len(train_dataloader)))

            if i_batch % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                now = datetime.now()
                current_time = now.strftime("%d.%m.%Y, %H:%M:%S")
                print_and_log(
                    f'{current_time}, Epoch {epoch - 1 + (float(i_batch) / len(train_dataloader)):02.2f}: train loss smoothed: {train_loss_mean:02.2f}, train loss batch: {train_loss:02.2f}, image loss: {loss_norm_image:02.2f}, text loss: {loss_norm_text:02.2f}, lr: {current_lr}',
                    log_fn)

        current_lr = optimizer.param_groups[0]['lr']
        now = datetime.now()
        current_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        print_and_log(
            f'{current_time}, Epoch {epoch - 1 + (float(i_batch) / len(train_dataloader)): 02.2f}: train loss smoothed: {train_loss_mean:02.2f}, train loss batch: {train_loss:02.2f}, image loss: {loss_norm_image:02.2f},text loss: {loss_norm_text:02.2f}, lr: {current_lr}',
            log_fn)

        # now every 5 epochs, validation loss may be interesting
        if epoch % 1 == 0:
            model = model.eval()
            test_loss = 0.
            clip.model.convert_weights(model)
            for (i_batch, (images, captions)) in enumerate(test_dataloader):
                with torch.no_grad():
                    # TODO: tokenize in dataloader for speed!
                    tokens = clip.tokenize(captions).to(device)
                    images = images.to(device)
                    logits_per_image, logits_per_text = model(images, tokens)
                    ground_truth = torch.arange(images.shape[0], dtype=torch.long).to(device)
                    loss_norm_image = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
                    loss_norm_text = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
                    test_loss += (loss_norm_image + loss_norm_text) / 2

            test_loss = test_loss / len(test_dataloader)
            tb_writer.add_scalar("validation loss", test_loss, epoch)

            save_fn(model, epoch, test_loss)

            print_and_log('======================================================================', log_fn)
            print_and_log('Epoch {:02.2f}: test loss: {:02.2f}'.format(epoch, test_loss), log_fn)
            print_and_log('======================================================================', log_fn)

            convert_models_to_fp32(model)
            model = model.train()
        tb_writer.flush()