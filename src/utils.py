
import pickle
import torch
import numpy as np
import os
import glob
from .models.EncoderProtoNet import EncoderProtoNet
from .Prototypical_Networks_for_Few_shot_Learning_PyTorch.src.prototypical_batch_sampler import PrototypicalBatchSampler
from .Prototypical_Networks_for_Few_shot_Learning_PyTorch.src.protonet import ProtoNet
from .Butterfly200DataSet import Butterfly200DataSet

# File for different utils functions used across the project

def load_dict(filename_):
    """
    loads a pickle saved object and return it 
    """
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
def save_dict(di_, filename_):
    """
    saves a pickle object in a certain path
    """
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
        
def init_seed(opt):
    torch.cuda.cudnn_enabled = True
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
def init_dataset(opt, mode,paths=None):
    """Initialize a dataset based on the options supplied

    Args:
        opt (arguments parser): options
        mode (str): train,val,test
        paths (list(str), optional): examples passed if no split dict is supplied in the options. Defaults to None.

    Raises:
        Exception: No paths or split_dict passed
        Exception: number of the examples per a certin class in the dataset can't satisfy the number of query + support examples

    Returns:
        Butterfly200DataSet: the dataset
    """
    if paths is not None:
        # paths are passed
        dataset = Butterfly200DataSet(paths=paths,precache=not opt.no_precache,mode=mode)
    elif opt.split_path is not None:
        # split dictionary and a mode are passed
        dataset = Butterfly200DataSet(split_dict_path=opt.split_path,precache=not opt.no_precache,mode=mode)
        n_classes = len(np.unique(dataset.y))
        if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
            raise(Exception('There are not enough classes in the dataset in order ' +
                            'to satisfy the chosen classes_per_it. Decrease the ' +
                            'classes_per_it_{tr/val} option and try again.'))
    else:
        raise Exception('a split dict or paths should be passed to the Butterfly200DataSet')
    
    return dataset


def init_sampler(opt, labels, mode):
    """Initializes a prototypical sampler

    Args:
        opt (arguments parser): options
        labels (list(int)): the dataset labels
        mode (str): train,val,test

    Returns:
        PrototypicalBatchSampler: A prototypical sampler
    """
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)
def init_dataloader(opt, mode='train',paths=None,no_proto_sampler=False,shuffle=False):
    """Initialize the dataloader

    Args:
        opt (arguments parser): options
        mode (str, optional): train,val,test. Defaults to 'train'.
        paths (list(str), optional): . Defaults to None.
        no_proto_sampler (bool, optional): Initialize the data loader with a normal batch sampler. Defaults to False.
        shuffle (bool, optional): To shuffle the data or not. Defaults to False.

    Returns:
        DataLoader: torch data loader
    """
    dataset = init_dataset(opt, mode,paths=paths)
    if  no_proto_sampler:
        # use a normal data loader
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size, shuffle=shuffle)
    else:
        sampler = init_sampler(opt, dataset.y, mode)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
        
    return dataloader
def init_protonet(opt,x_dim=128):
    """Initialize the ProtoNet

    Args:
        opt (arguments parser): options
        x_dim (int, optional): number of channels in the first conv block. Defaults to 128.

    Returns:
        ProtoNet: ProtoNet model
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    
    model = ProtoNet(x_dim=x_dim).to(device)
    return model
def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)
def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)
def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)
def init_hybrid_model(opt):
    """Initialize the hyprid res-proto model

    Args:
        opt (arguments parser): options

    Returns:
        EncoderProtoNet: The hyprid model
    """
    model = EncoderProtoNet(proto_x_dim=128)
    if opt.hybrid_model_weights:
        model.load_state_dict(torch.load(opt.hybrid_model_weights))
    if opt.enc_weights:
        model.load_encoder_weights(opt.enc_weights)
    if opt.proto_weights:
        model.load_proto_weights(opt.proto_weights)
        
    if opt.proto_weights and not opt.enc_weights:
        print('warning: you are trying to initialize the protonet weights without initializing the resnet weights')
    
    model.encoder.requires_gradient = not opt.freeze_encoder if hasattr(opt,'freeze_encoder') else False
    model.proto.requires_gradient = not opt.freeze_proto if hasattr(opt,'freeze_proto') else False
    model = model.to('cuda')
    return model
def get_files(filepath,expression='*.json'):
    '''
    Walks over a directory and its children to get all children json files pathes
    Arguments:
    file_path: string that specifies the path to the data parent directory 
    Returns:
    all_files: List of all the filepaths of the matching expression files included in the directory
    '''
    all_files = []
    for root, dirs, files in os.walk(filepath):
        files = glob.glob(os.path.join(root,expression))
        for f in files :
            all_files.append(os.path.abspath(f))
    return all_files
def classify_vector_with_means(out,means):
    """Classifies the provided vectors with the classes means
    using eculidean distance 

    Args:
        out (_type_): the vector of the required sample to be classified
        means (numpy_array(numpy_array(double))): the list of the classes means n-dimensional vectors

    Returns:
        best_class (int): The number of the best class matched to the out vector
        min_d (double) : The distance between the mean of the best class and "out" 
    """
    min_d = 10**6
    best_class = -1
    out = np.array(out)
    
    for key in means:
        d = np.sqrt(np.sum((out-np.array(means[key]))**2))
        if d<min_d:
            best_class = key
            min_d = d

    return best_class,min_d
def get_classes_samples(files_paths):
    """Maps the classes numbers to their samples paths based on the butterfly 200 dataset naming 

    Args:
        files_paths (list(str)): List of all paths

    Returns:
        dict: A dict that maps the classes numbers to their samples paths 
    """
    classes_samples = {}
    for path in files_paths:
        class_num = int(os.path.basename(path.split('.')[0]))-1
        if class_num not in classes_samples:
            classes_samples[class_num] = []
        classes_samples[class_num].append(path)
    return classes_samples