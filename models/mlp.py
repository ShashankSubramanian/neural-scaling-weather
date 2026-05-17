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
