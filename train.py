import os
import torch
import numpy as np
from src.parsers.train_parser_util import get_train_parser
from src.utils import save_list_to_file,init_dataloader,init_hybrid_model,init_optim,init_lr_scheduler
from src.Prototypical_Networks_for_Few_shot_Learning_PyTorch.src.prototypical_loss import prototypical_loss as loss_fn
from tqdm import tqdm
def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'


    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    # prepare the model's path for saving
    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)

            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            torch.cuda.empty_cache()
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in tqdm(val_iter):
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            torch.cuda.empty_cache()
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc
def train_main(opt):
    if not os.path.exists(opt.experiment_root):
        os.makedirs(opt.experiment_root)
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
    print('Initializing the training data loader')
    tr_dataloader = init_dataloader(opt, 'train')
    print('Initializing the validation data loader')
    val_dataloader = init_dataloader(opt, 'val')
    
    model = init_hybrid_model(opt)
    optim = init_optim(opt, model)
    lr_scheduler = init_lr_scheduler(opt, optim)
    print('start training')
    res = train(opt=opt,
            tr_dataloader=tr_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optim=optim,
            lr_scheduler=lr_scheduler)
if __name__ == "__main__":
    opt = get_train_parser().parse_args()
    train_main(opt)