import torch
import torch.nn as nn

# networks
from models.swin import SwinTransformer
from collections import OrderedDict


def _strip_ddp_prefix(state_dict):
    """Remove the DistributedDataParallel prefix from a model state dict."""
    if all(key.startswith("module.") for key in state_dict):
        return OrderedDict(
            (key.removeprefix("module."), value)
            for key, value in state_dict.items()
        )
    return OrderedDict(state_dict)


def load_legacy_model(model, state_dict):
    """Load a hybrid Transformer Engine Swin checkpoint into the current model.

    Legacy checkpoints used Transformer Engine linear layers, but retained the
    original NormLayer and MLP module hierarchy. Current Swin blocks use TE
    RMSNorm and LayerNormMLP modules throughout, so their parameter names differ
    even though the parameter shapes and computations are compatible.
    """
    state_dict = _strip_ddp_prefix(state_dict)
    converted_state = OrderedDict()

    key_replacements = (
        (".norm1.norm.weight", ".norm1.weight"),
        (".attn.q_norm.norm.weight", ".attn.q_norm.weight"),
        (".attn.k_norm.norm.weight", ".attn.k_norm.weight"),
        (".norm2.norm.weight", ".rms_norm_mlp.layer_norm_weight"),
        (".mlp.fc1.weight", ".rms_norm_mlp.fc1_weight"),
        (".mlp.fc1.bias", ".rms_norm_mlp.fc1_bias"),
        (".mlp.fc2.weight", ".rms_norm_mlp.fc2_weight"),
        (".mlp.fc2.bias", ".rms_norm_mlp.fc2_bias"),
    )

    for key, value in state_dict.items():
        # TE metadata is version-specific and contains no learned parameters.
        if key.endswith("._extra_state"):
            continue
        for old, new in key_replacements:
            if old in key:
                key = key.replace(old, new)
                break
        converted_state[key] = value

    # Initialize version-specific TE metadata from the current modules.
    for key, value in model.state_dict().items():
        if key.endswith("._extra_state"):
            converted_state[key] = value

    model.load_state_dict(converted_state, strict=True)
    return model


def load_model_state(model, state_dict):
    """Load a current-format model state dict, accepting an optional DDP prefix."""
    model.load_state_dict(_strip_ddp_prefix(state_dict), strict=True)
    return model


class TimeStepper(nn.Module):
    """Wrapper for time stepping during training"""

    def __init__(self, cfg, model_handle):
        super(TimeStepper, self).__init__()
        self.model = model_handle
        self.num_rollout_steps = cfg.train.num_rollout_steps
        self.num_invariants = len(cfg.data.invariants)

    def forward(self, inp):
        result = []
        inpt = inp
        c_in = inp.shape[2]

        # stepper
        for step in range(self.num_rollout_steps):
            pred = self.model(inpt)
            c_out = pred.shape[2] # can be different if there are invariants
            result.append(pred)
            if step == self.num_rollout_steps:
                break
            # add back invariants at every step
            n_invariants = (c_in - c_out)
            assert (n_invariants == self.num_invariants), "number of invariants does not match"
            invars = inp[:, :, -n_invariants:, :, :]
            inpt = torch.cat([pred, invars], dim=2)

        result = torch.cat(result, dim=1)
        return result


def get_model(cfg, domain_metadata=None):
    name = cfg.model.arch
    if name == "swin":
        model = SwinTransformer.instantiate_from_cfg(cfg, domain_metadata=domain_metadata)
    else:
        raise NotImplementedError(f"model type {name} not implemented")

    # wrap the model into a time stepper to deal with
    # multistep finetuning etc
    model = TimeStepper(cfg, model)
    return model


def load_model(model, checkpoint_file, local_rank):
    map_location = "cuda:{}".format(local_rank) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        checkpoint_file,
        map_location=map_location,
        weights_only=False,
    )
    return load_model_state(model, checkpoint["model_state"])
