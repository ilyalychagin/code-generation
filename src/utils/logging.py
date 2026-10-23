import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch


def validate_logging_config(config):
    if config.log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be positive")
    if config.comet_upload_timeout < 1:
        raise ValueError("comet_upload_timeout must be positive")
    if config.log_code_every_n_steps is not None and config.log_code_every_n_steps < 1:
        raise ValueError("log_code_every_n_steps must be positive or None")
    if config.log_code_num_samples < 1:
        raise ValueError("log_code_num_samples must be positive")


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

    from comet_ml import OfflineExperiment

    experiment = OfflineExperiment(
        offline_directory=str(output_dir / "comet"),
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


def upload_comet_archive(archive, timeout=60):
    command = [sys.executable, "-m", "comet_ml.scripts.comet_upload", str(archive)]
    if not os.environ.get("COMET_API_KEY"):
        print("Comet upload skipped: COMET_API_KEY is not set.", flush=True)
        print(f"Upload later: {shlex.join(command)}", flush=True)
        return

    print(f"Uploading Comet archive: {archive}", flush=True)
    try:
        subprocess.run(command, check=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as error:
        print(f"Comet upload failed; local archive kept: {archive}. Reason: {error}", flush=True)
        print(f"Upload later: {shlex.join(command)}", flush=True)
    else:
        print(f"Comet upload completed; local archive kept: {archive}", flush=True)


def finish_experiment(experiment, config):
    if experiment is None:
        return
    try:
        experiment.end()
    except Exception as error:
        print(f"Could not finalize Comet offline logs: {error}", flush=True)
        return

    archive = Path(experiment.offline_directory) / f"{experiment.get_key()}.zip"
    print(f"Comet offline archive: {archive}", flush=True)
    if config.comet_upload_on_end:
        upload_comet_archive(archive, timeout=config.comet_upload_timeout)


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


@torch.inference_mode()
def log_code_samples(
    model, step, epoch, *, experiment, tokenizer, samples, collate_fn,
    device, batch_size, max_new_tokens, num_beams=1,
):
    was_training = model.training
    model.eval()
    rows = []
    try:
        for split_name, examples in samples.items():
            for start in range(0, len(examples), batch_size):
                batch_examples = examples[start:start + batch_size]
                batch = {
                    key: value.to(device)
                    for key, value in collate_fn(batch_examples).items()
                }
                output_tokens = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                )
                questions = tokenizer.batch_decode(
                    batch["input_ids"].cpu(), skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                predictions = tokenizer.batch_decode(
                    output_tokens.cpu(), skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for index, (example, question, prediction) in enumerate(
                    zip(batch_examples, questions, predictions), start=start
                ):
                    reference = (
                        tokenizer.decode(
                            example["label_ids"], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        if "label_ids" in example else ""
                    )
                    rows.append([
                        step, epoch, split_name, index, question, reference, prediction,
                    ])
    finally:
        model.train(was_training)

    experiment.log_table(
        filename=f"code_samples_step_{step}.csv",
        tabular_data=rows,
        headers=["step", "epoch", "split", "sample", "question", "reference", "prediction"],
    )
