import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

def RotatE_Trans(ent, rel, is_hr):
    re_ent, im_ent = ent
    re_rel, im_rel = rel
    if is_hr:   #ent == head
        re = re_ent * re_rel - im_ent * im_rel
        im = re_ent * im_rel + im_ent * re_rel
    else:   #ent == tail 
        re = re_rel * re_ent + im_rel * im_ent
        im = re_rel * im_ent - im_rel * re_ent
    return re, im
