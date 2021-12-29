import os
import torch
import torch.nn as nn


def save_model(model, saved_dir, name):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {}
    if isinstance(model, nn.DataParallel):
        check_point["model_state_dict"] = model.module.state_dict()
    else:
        check_point["model_state_dict"] = model.state_dict()
    torch.save(check_point, saved_dir + "/best_model_weight_{}.pt".format(name))


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(
    model,
    saved_dir,
    use_cuda=True,
    eval=True,
):
    if use_cuda:
        device = "cuda"
        net = model.to("cuda")
    else:
        device = "cpu"
        net = model.to("cpu")
    cp = torch.load(saved_dir, map_location=device)
    if isinstance(net, nn.DataParallel):
        net.module.load_state_dict(cp["model_state_dict"], strict=False)
    else:
        new_dict = remove_prefix(cp["model_state_dict"], "module.")
        net.load_state_dict(new_dict, strict=False)
    if eval:
        net.eval()
    return net


if __name__ == "__main__":
    pass
