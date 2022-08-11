import torch
from torch import nn
from .SubResNet import SubResNet18
from .ProtoNet import ProtoNet
class EncoderProtoNet(nn.Module):
    def __init__(self,proto_x_dim=3, proto_hid_dim=64, proto_z_dim=64):
        super(EncoderProtoNet, self).__init__()
        self.encoder = SubResNet18()
        self.proto = ProtoNet(proto_x_dim,proto_hid_dim,proto_z_dim)
    def load_encoder_weights(self,path,strict=False):
        self.encoder.load_state_dict(torch.load(path),strict=strict)
    def load_proto_weights(self,path,strict=False):
        self.proto.load_state_dict(torch.load(path),strict=strict)
    def set_encoder_model(self,model):
        self.encoder = model
    def set_proto_model(self,model):
        self.proto = model
    def forward(self,x):
        x = self.encoder(x)
        x = self.proto(x)
        return x