import torch
import torch.nn as nn

# networks
from models.swin import SwinTransformer
from collections import OrderedDict


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
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError:
        if not all(key.startswith("module.") for key in checkpoint["model_state"]):
            raise
        new_state_dict = OrderedDict()
        for key, val in checkpoint["model_state"].items():
            name = key.removeprefix("module.")
            new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    return model
