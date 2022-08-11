from src.parsers.classes_means_parser import get_classes_means_parser
from src.utils import init_dataloader, init_hybrid_model,save_dict,load_dict,get_classes_samples
import numpy as np
from tqdm import tqdm

def construct_means_main(opt):
    model = init_hybrid_model(opt)
    model.eval()
    # dictionary - key is the class number - values are the paths to the preprocessed 
    # tensors (or the images) to be used to calculate the means 
    split_path = opt.split_path
    split_d = load_dict(split_path)
    # use the training values to calculate the means
    classes_paths_d = get_classes_samples(split_d['train'])
    d = {}
    # calculate medians instead of the means
    if opt.use_median:
        for cls in tqdm(classes_paths_d):
            data_loader = init_dataloader(opt,paths=classes_paths_d[cls],no_proto_sampler=True)
            ks = []
            num = 0
            for batch in data_loader:
                x,_ = batch
                out = model(x.to('cuda'))
                out = out.detach().cpu().numpy()
                ks.extend(out)
            
            d[cls] = np.median(ks,axis=0)
    else:
        # calculate the means
        for cls in tqdm(classes_paths_d):
            data_loader = init_dataloader(opt,paths=classes_paths_d[cls],no_proto_sampler=True)
            ks = []
            num = 0
            for batch in data_loader:
                x,_ = batch
                out = model(x.to('cuda'))
                out = out.detach().cpu().numpy()
                ks.append(np.sum(out,axis=0))
                num += out.shape[0]
            ks = np.array(ks)
            
            d[cls] = np.sum(ks,axis=0)/num
    
    save_dict(d,opt.save_path)
    print('saved at '+opt.save_path)
if __name__ == "__main__":
    opt = get_classes_means_parser().parse_args()
    construct_means_main(opt)
    
    
    
            
            
    