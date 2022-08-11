import torch.nn as nn
def conv_block(in_channels, out_channels,use_dropout=False):
    '''
    returns a block conv-bn-relu-pool
    '''
    start_part = [nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()]
    
    end_part = [nn.MaxPool2d(2)]
    drop_part = []
    if use_dropout:
        drop_part = [nn.Dropout(p=0.1)]
    final_part = start_part+drop_part+end_part
    return nn.Sequential(*final_part)



class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
