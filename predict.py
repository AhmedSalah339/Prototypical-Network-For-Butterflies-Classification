import os
from src.utils import init_dataloader, init_hybrid_model,classify_vector_with_means,load_dict,get_files
from src.parsers.predict_parser import get_predict_parser
import pandas as pd
if __name__ == '__main__':
    opt = get_predict_parser().parse_args()
    
    path = opt.examples_path
    if not os.path.isdir(path):
        raise Exception('You should provide a directory for the examples to be predicted')
    
    # get the list of the images paths
    files_paths = get_files(path,opt.suffix)
    
    model = init_hybrid_model(opt)
    model.eval()
    
    data_loader = init_dataloader(opt,paths=files_paths,no_proto_sampler=True,mode='val')
    
    means = load_dict(opt.means_path)
    d = {
        'paths':[],
        'predictions':[]  
    }
    i=0
    for batch in data_loader:
        clss = []
        x,_ = batch
        out = model(x.to('cuda'))
        out = out.detach().cpu().numpy()
        for o in out:
            # classify each vector using the classes centers
            cls,_ = classify_vector_with_means(o,means)
            clss.append(cls)
        d['paths'].extend(files_paths[i:i+opt.batch_size])
        d['predictions'].extend(clss)
        i += opt.batch_size
    df = pd.DataFrame(d)
    df.to_csv(opt.save_path)

    
    