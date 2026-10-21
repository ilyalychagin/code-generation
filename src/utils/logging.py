import json
import os
from dataclasses import asdict
from pathlib import Path

import torch


def validate_logging_config(config):
    if config.log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be positive")
    if config.comet_enabled and not os.environ.get("COMET_API_KEY"):
        raise ValueError("Set COMET_API_KEY before enabling Comet logging")


def _flatten_parameters(values, prefix=""):
    parameters = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            parameters.update(_flatten_parameters(value, key))
        elif isinstance(value, (tuple, list)):
            parameters[key] = json.dumps(value, default=str)
        elif isinstance(value, Path):
            parameters[key] = str(value)
        else:
            parameters[key] = value
    return parameters


def create_experiment(config, model, optimizer, scheduler, device):
    """Save run parameters locally and optionally start a Comet experiment."""
    validate_logging_config(config)
    parameters = {
        "config": asdict(config),
        "model": model.config.to_dict(),
        "optimizer": {
            "name": type(optimizer).__name__,
            "defaults": optimizer.defaults,
            "groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ],
        },
        "scheduler": {
            "name": type(scheduler).__name__,
            **{
                name: getattr(scheduler, name)
                for name in (
                    "mode", "factor", "patience", "threshold",
                    "threshold_mode", "cooldown", "min_lrs", "eps",
                )
                if hasattr(scheduler, name)
            },
        },
        "runtime": {
            "device": str(device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2, ensure_ascii=False, default=str)

    if not config.comet_enabled:
        return None

    from comet_ml import Experiment

    experiment = Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name=config.comet_project,
        workspace=config.comet_workspace,
        auto_metric_logging=False,
        auto_param_logging=False,
        parse_args=False,
        auto_output_logging=None,
        log_code=False,
        log_graph=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        log_env_gpu=False,
        log_env_cpu=False,
        log_env_host=False,
        log_env_network=False,
        log_env_disk=False,
        auto_log_co2=False,
        auto_histogram_weight_logging=False,
        auto_histogram_gradient_logging=False,
        auto_histogram_activation_logging=False,
    )
    try:
        experiment.set_name(config.comet_experiment_name)
        experiment.log_parameters(_flatten_parameters(parameters))
    except BaseException:
        experiment.end()
        raise
    return experiment


@torch.no_grad()
def gradient_norm(model):
    """Global L2 norm of the current gradients, without changing them."""
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return torch.stack(norms).norm(2).item()
