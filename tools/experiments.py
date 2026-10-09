"""Enumerate or launch independent, logged manuscript study configurations."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MAIN = ("cifar100", "tiny_imagenet", "imagenette", "imagewoof", "imagenet1k")
CROSS = ("efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x0_5", "swin_t",
         "wide_resnet50_2", "densenet121", "densenet169", "densenet201")
COMPONENTS = {
    "fadrm": {"assignment": "static", "topology": "none"},
    "drc": {"assignment": "independent", "topology": "none"},
    "pta": {"assignment": "static", "topology": "pta"},
    "stc": {"assignment": "stc", "topology": "none"},
    "hfta": {"assignment": "static", "topology": "hfta"},
    "reta": {"assignment": "independent", "topology": "pta"},
    "drc_hfta": {"assignment": "independent", "topology": "hfta"},
    "stc_pta": {"assignment": "stc", "topology": "pta"},
    "retapp": {},
}


def cases(study):
    if study == "main":
        return [(d, k, {}, ("resnet18", "resnet50", "resnet101"), "main") for d in MAIN for k in (1, 10, 50)]
    if study == "high-ipc":
        return [(d, k, {}, ("resnet18",), "high") for d, ks in
                (("cifar10", (250, 500, 1000, 1500)), ("cifar100", (100, 150)), ("tiny_imagenet", (100,))) for k in ks]
    if study == "cross":
        return [("imagenet1k", 10, {}, CROSS, "cross")]
    if study == "components":
        return [(d, 10, values, ("resnet18",), name) for d in
                ("cifar100", "tiny_imagenet", "imagenette", "imagenet1k") for name, values in COMPONENTS.items()]
    if study == "assignment":
        return [(d, 10, values, ("resnet18",), name) for d in ("imagenet1k", "tiny_imagenet")
                for name, values in (("independent", {"assignment": "independent"}),
                                     ("balanced", {"assignment": "balanced"}),
                                     ("capacity_no_structure", {"structure_weight": 0}), ("stc", {}))]
    if study == "packed":
        return [(d, 1, values, ("resnet18",), name) for d in MAIN for name, values in
                (("whole_image", {"cell_slots": False, "cell_replay": False}),
                 ("cells_no_replay", {"cell_replay": False}),
                 ("cells_replay", {}), ("cells_static", {"assignment": "static"}),
                 ("cells_no_topology", {"topology": "none"}))]
    if study == "stc-sweep":
        return [("imagenet1k", 10, {"complexity_weight": cp, "structure_weight": st}, ("resnet18",), f"cp{cp}_st{st}")
                for cp in (0., .1, .2, .3, .4) for st in (0., .1, .2, .3, .4, .5, .6)]
    if study == "hfta-sweep":
        return [(d, k, {"hfta_weight": weight}, ("resnet18",), f"hfta{weight}")
                for d in ("tiny_imagenet", "imagenet1k") for k in (1, 10) for weight in (0., .1, .3, .5, .7, 1.)]
    if study == "group-pool":
        return [("imagenet1k", 10, {"group_size": group, "candidates": pool}, ("resnet18",), f"g{group}_m{pool}")
                for group, pool in ((2, 32), (4, 32), (8, 32), (4, 16), (4, 64))]
    raise ValueError(study)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("study", choices=("main", "high-ipc", "cross", "components", "assignment", "packed", "stc-sweep", "hfta-sweep", "group-pool"))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--teachers", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("outputs"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    for dataset, ipc, overrides, students, variant in cases(a.study):
        for seed in a.seeds:
            output = a.output_root/f"{a.study}_{variant}_{dataset}_ipc{ipc}_seed{seed}"
            command = [sys.executable, "-u", str(ROOT/"run.py"), "all", "--dataset", dataset,
                       "--ipc", str(ipc), "--seed", str(seed), "--data", str(a.data_root/dataset),
                       "--teacher-checkpoint", str(a.teachers/dataset/"resnet18.pth"),
                       "--output", str(output), "--students", *students]
            import json
            for key, value in overrides.items():
                command += ["--set", f"{key}={json.dumps(value)}"]
            print(shlex.join(command), flush=True)
            if a.execute:
                subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
