"""Command-line entry points for the recover/relabel/validate protocol."""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import Config, DATASETS, PRESETS, STUDENTS
from .data import ImageSet, RealView, Source, save_json, split_digest, split_indices
from .models import Teacher, build_model, load_weights
from .recovery import recover_class
from .runtime import Log, file_hash, seed_all
from .training import accuracy, relabel, select_preset, train_student


def read_run(output):
    metadata = json.loads((Path(output)/"run.json").read_text())
    return Config(**metadata["config"]).validate(), metadata


def make_teacher(config, path, device):
    model = build_model(config.teacher, config.classes, config.size)
    load_weights(model, path)
    return Teacher(model.to(device), config.dataset, config.size, config.first_bn_multiplier)


def loader(source, indices, config, workers, augment=False):
    return DataLoader(RealView(source, indices, config.size, augment),
                      batch_size=128, shuffle=augment, num_workers=workers)


def configuration(args):
    overrides = {}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("--set takes key=value")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        overrides[key] = value
    for key in ("dataset", "ipc", "seed", "teacher"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return Config.read(args.config, **overrides)


def recover(args):
    config = configuration(args)
    seed_all(config.seed)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return
    if args.data is None or args.teacher_checkpoint is None:
        raise ValueError("Recovery requires --data and --teacher-checkpoint")
    source = Source(args.data, config.dataset)
    if len(source.classes) != config.classes:
        raise ValueError(f"Expected {config.classes} classes, found {len(source.classes)}")
    split = split_indices(source.labels, config.dataset, config.split_seed)
    metadata = {"config": config.to_dict(), "classes": source.classes,
                "data_root": str(args.data.resolve()), "split_sha256": split_digest(split),
                "teacher_sha256": file_hash(args.teacher_checkpoint)}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    existing = output/"run.json"
    if existing.exists() and json.loads(existing.read_text()) != metadata:
        raise ValueError("Run directory belongs to different data, teacher, or configuration")
    save_json(existing, metadata)
    save_json(output/"split.json", split)
    teacher = make_teacher(config, args.teacher_checkpoint, args.device)
    log = Log(output/"logs/recovery.output.txt")
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        end = config.classes if args.class_end is None else min(args.class_end, config.classes)
        for cls in range(args.class_start, end):
            if (output/"synthetic"/f"{cls:05d}"/"complete.json").exists():
                log(stage="recover", cls=cls, skipped="complete")
                continue
            recover_class(source, split["train"][str(cls)], teacher, config, cls, output, log)
        if torch.device(args.device).type == "cuda":
            save_json(output/"recovery_memory.json", {"peak_bytes": torch.cuda.max_memory_allocated()})
    finally:
        teacher.close()
        log.close()


def relabel_run(args):
    config, metadata = read_run(args.output)
    if args.teacher_checkpoint is None or file_hash(args.teacher_checkpoint) != metadata["teacher_sha256"]:
        raise ValueError("Relabeling must use the recovery teacher checkpoint")
    images = ImageSet(args.output/"synthetic")
    if len(images) != config.classes*config.ipc:
        raise ValueError("Complete recovery for every class before relabeling")
    teacher = make_teacher(config, args.teacher_checkpoint, args.device)
    log = Log(args.output/"logs/relabel.output.txt")
    try:
        relabel(images, teacher, config, args.output/"labels", log)
    finally:
        teacher.close()
        log.close()


def train_run(args):
    config, metadata = read_run(args.output)
    source = Source(args.data or metadata["data_root"], config.dataset)
    split = json.loads((args.output/"split.json").read_text())
    if source.classes != metadata["classes"]:
        raise ValueError("Development labels do not match the recovery class order")
    indices = [i for group in split["development"].values() for i in group]
    development = loader(source, indices, config, args.workers)
    images = ImageSet(args.output/"synthetic")
    for arch in args.students:
        for preset in args.presets:
            destination = args.output/"students"/arch/preset
            if (destination/"metrics.json").exists():
                continue
            log = Log(args.output/"logs"/f"{arch}_{preset}.output.txt")
            try:
                train_student(images, config, args.output/"labels", development,
                              arch, preset, destination, args.device, log)
            finally:
                log.close()


def evaluate_run(args):
    config, metadata = read_run(args.output)
    source = Source(args.data or metadata["data_root"], config.dataset, train=False)
    if source.classes != metadata["classes"]:
        raise ValueError("Test folder must use the same class order as training")
    test = loader(source, range(len(source)), config, args.workers)
    for arch in args.students:
        checkpoint, chosen = select_preset(args.output/"students", arch)
        model = build_model(arch, config.classes, config.size).to(args.device)
        load_weights(model, checkpoint)
        result = {**chosen, "test_accuracy": accuracy(model, test, config.dataset, args.device)}
        save_json(args.output/"results"/f"{arch}.json", result)
        print(json.dumps(result), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RETA++ recovery and cached-label student training")
    parser.add_argument("stage", choices=("recover", "relabel", "train", "evaluate", "all"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/paper.json")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--ipc", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--teacher", choices=STUDENTS)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=["resnet18"])
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--class-start", type=int, default=0)
    parser.add_argument("--class-end", type=int)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    if args.print_config:
        print(json.dumps(configuration(args).to_dict(), indent=2))
        return
    functions = {"recover": recover, "relabel": relabel_run,
                 "train": train_run, "evaluate": evaluate_run}
    for stage in functions if args.stage == "all" else [args.stage]:
        functions[stage](args)
