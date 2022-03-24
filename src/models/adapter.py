""" Adapters """
import collections
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
import pdb


##### Adapter #####
class AdapterConfig(NamedTuple):
    input_size: int
    hidden_size: int
    act: str
    init_range: float


class Adapter(nn.Module):

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.down_project = nn.Linear(config.input_size, config.hidden_size)
        nn.init.normal_(self.down_project.weight, std=config.init_range)
        nn.init.zeros_(self.down_project.bias)

        self.activation = ACT2FN[config.act]

        self.up_project = nn.Linear(config.hidden_size, config.input_size)
        nn.init.normal_(self.up_project.weight, std=config.init_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected


##### Insert Adapter #####


class AdaBartLayerFF(nn.Module):

    def __init__(self, ff: nn.Linear, config: AdapterConfig):
        super().__init__()
        self.ff = ff
        self.adapter = Adapter(config).to(ff.weight.device)

    def forward(self, hidden_states):
        return self.adapter(self.ff(hidden_states))


def _ada_bart_layer_ff(config: AdapterConfig):
    return lambda ff: AdaBartLayerFF(ff, config=config)

def insert_adapters(args, model):

    adapter_config = AdapterConfig(input_size=model.config.d_model,
                                   hidden_size=args.adapter_size,
                                   act=args.adapter_act,
                                   init_range=args.adapter_init_range)

    # Record modules to be replaced
    replace_module_names = []
    ModuleName = collections.namedtuple("ModuleName", "parent, child")
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, nn.Linear):
            name_split = name.split(".")
            if name_split[-1] in ['fc2', 'out_proj']:
                replace_module_names.append(
                    ModuleName(".".join(name_split[:-1]), name_split[-1]))

    # Replace modules accord to names
    for parent_name, child_name in replace_module_names:
        for name, sub_module in model.named_modules():
            if name == parent_name:
                setattr(
                    sub_module, child_name,
                    _ada_bart_layer_ff(adapter_config)(getattr(
                        sub_module, child_name)))
                break

    return model
