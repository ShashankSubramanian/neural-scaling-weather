import torch
import torch.nn as nn
from utils import comm
from models.layer_helpers import (
    LinearLayer,
    BiasLayer,
)
from distributed.mappings import (
    identity_forward_allreduce_backward,
    allreduce_forward_identity_backward,
)
from utils.profiler_utils import profile

import transformer_engine.pytorch as te

class MLPTransformerEngine(nn.Module):
    """MLP with TE"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if comm.get_size("tp") > 1:
            tp_group = comm.get_group("tp")
            tp_size = comm.get_size("tp")
        else:
            tp_group = None
            tp_size = 1

        self.fc1 = te.Linear(
            in_features,
            hidden_features,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.act = act_layer()
        self.fc2 = te.Linear(
            hidden_features,
            out_features,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="row" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.drop = nn.Dropout(drop)

        # mark weights
        self.fc1.weight.comm_metadata = {
            "sharded": ["tp", None],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.fc1.bias.comm_metadata = {
            "sharded": ["tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.fc2.weight.comm_metadata = {
            "sharded": [None, "tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.fc2.bias.comm_metadata = {
            "sharded": [],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }

    @profile("mlp")
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP(nn.Module):
    """MLP as used in Transformers
    and includes tensor+sequence parallelism
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # compute local features
        comm_tp_size = comm.get_size("tp")
        assert (
            hidden_features % comm_tp_size == 0
        ), f"cannot shard MLP hidden {hidden_features} into {comm_tp_size} shards"
        hidden_features_local = hidden_features // comm_tp_size

        self.fc1 = LinearLayer(
            in_features,
            hidden_features_local,
            comm_metadata={
                "sharded": ["tp", None],
                "shared": ["sp1-sp2"],
                "reduce": ["sp1-sp2"],
            },
        )
        self.bias1 = BiasLayer(
            hidden_features_local,
            comm_metadata={
                "sharded": ["tp"],
                "shared": ["sp1-sp2"],
                "reduce": ["sp1-sp2"],
            },
        )
        self.act = act_layer()
        self.fc2 = LinearLayer(
            hidden_features_local,
            out_features,
            comm_metadata={
                "sharded": [None, "tp"],
                "shared": ["sp1-sp2"],
                "reduce": ["sp1-sp2"],
            },
        )
        self.bias2 = BiasLayer(
            out_features,
            comm_metadata={
                "sharded": [],
                "shared": ["tp-sp1-sp2"],
                "reduce": ["sp1-sp2"],
            },
        )
        self.drop = nn.Dropout(drop)

    @profile("mlp")
    def forward(self, x):
        # tp comms
        x = identity_forward_allreduce_backward(x, comm_name="tp")

        x = self.fc1(x)
        x = self.bias1(x)  # note this bias is sharded

        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)

        # tp comms
        x = allreduce_forward_identity_backward(x, comm_name="tp")

        # add after allreduce so you don't add it too many times
        x = self.bias2(x)  # note this bias is shared in tp as well
        x = self.drop(x)

        return x
