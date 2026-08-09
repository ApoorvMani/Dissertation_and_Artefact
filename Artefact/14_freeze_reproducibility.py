"""
freeze the current dissertation environment and evidence.

this does not rerun any experiment.
it only records:
- python/environment versions
- cuda/gpu information
- pip freeze
- sha256 hashes of important scripts/results
"""

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


# --------------------------------------------------------------- config

OUT = Path("audit/reproducibility")

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# files that define the current experiment
SCRIPT_FILES = [
    "01_download_assets.py",
    "02_build_datasets.py",
    "02_clean_baseline.py",
    "04_tokenbreak_attack.py",
    "05_advtok_attack.py",
    "07_unicode_attacks.py",
    "08_defences.py",
    "09_defence_matrix.py",
    "10_analysis.py",
    "11_complex_utility.py",
    "12_cross_tokenizer_baseline.py",
    "13_freeze_current_results.py",
    "attack_schema.py",
    "requirements.txt",
]


# main empirical evidence
DATA_FILES = [
    "data/jigsaw_toxic_250.csv",
    "data/jigsaw_benign_250.csv",

    "data/clean_baseline_toxic.csv",
    "data/clean_baseline_benign.csv",

    "data/tokenbreak_results.csv",
    "data/advtok_results.csv",

    "data/unicode_invisible.csv",
    "data/unicode_homoglyph.csv",
    "data/unicode_compat.csv",
    "data/unicode_reorder.csv",
    "data/unicode_results.csv",

    "data/cpt_thresholds.json",

    "data/defence_matrix.csv",
    "data/defence_matrix_summary.csv",

    "data/complex_legitimate_600.csv",

    "data/cross_tokenizer_eligibility.csv",
    "data/cross_tokenizer_shared_eligible.csv",
    "data/cross_tokenizer_metadata.json",
]


# final analysis evidence
RESULT_FILES = [
    "results/step10/attack_metrics.csv",
    "results/step10/clean_controls.csv",
    "results/step10/clean_summary.csv",
    "results/step10/detection_metrics.csv",

    "results/step11/complex_utility_rows.csv",
    "results/step11/complex_utility_summary.csv",
    "results/step11/complex_utility_overall.csv",

    "results/final_current/coverage_matrix.csv",
    "results/final_current/margin_stratified_asr.csv",
    "results/final_current/margin_overall.csv",
    "results/final_current/headline_summary.txt",
]


# audit evidence
AUDIT_FILES = [
    "audit/AUDIT_CONTEXT.md",
    "audit/FILE_MAP.md",
    "audit/EXPERIMENT_REGISTRY.csv",
    "audit/EXPERIMENT_REGISTRY.md",
    "audit/AUDIT_WARNINGS.md",
]


ALL_FILES = (
    SCRIPT_FILES
    + DATA_FILES
    + RESULT_FILES
    + AUDIT_FILES
)


# --------------------------------------------------------------- helpers

def package_version(name):
    """return installed package version."""
    try:
        return version(name)

    except PackageNotFoundError:
        return "NOT INSTALLED"


def sha256_file(path):
    """sha256 a file without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def run_command(command):
    """run a command and return its text output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return (
                "COMMAND FAILED\n"
                + result.stderr.strip()
            )

        return result.stdout.strip()

    except Exception as exc:
        return (
            "COMMAND FAILED: "
            + str(exc)
        )


# --------------------------------------------------------------- environment

def collect_environment():
    """collect the environment we are running in now."""

    info = {
        "snapshot_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "python_executable":
            sys.executable,

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "machine":
            platform.machine(),

        "processor":
            platform.processor(),

        "packages": {
            "torch":
                package_version("torch"),

            "transformers":
                package_version(
                    "transformers"
                ),

            "tokenizers":
                package_version(
                    "tokenizers"
                ),

            "numpy":
                package_version("numpy"),

            "pandas":
                package_version("pandas"),

            "scipy":
                package_version("scipy"),

            "matplotlib":
                package_version(
                    "matplotlib"
                ),

            "datasets":
                package_version(
                    "datasets"
                ),

            "huggingface_hub":
                package_version(
                    "huggingface-hub"
                ),
        },
    }

    # torch information
    try:
        import torch

        info["torch"] = {
            "version":
                torch.__version__,

            "cuda_available":
                torch.cuda.is_available(),

            "cuda_version":
                torch.version.cuda,

            "cudnn_version":
                (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),

            "device_count":
                torch.cuda.device_count(),
        }

        devices = []

        for i in range(
            torch.cuda.device_count()
        ):
            props = (
                torch.cuda.get_device_properties(
                    i
                )
            )

            devices.append(
                {
                    "index":
                        i,

                    "name":
                        props.name,

                    "total_memory_bytes":
                        props.total_memory,

                    "compute_capability":
                        (
                            f"{props.major}."
                            f"{props.minor}"
                        ),
                }
            )

        info["torch"]["devices"] = (
            devices
        )

    except Exception as exc:
        info["torch"] = {
            "error": str(exc)
        }

    # independent gpu/driver view
    info["nvidia_smi"] = run_command(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,"
            "driver_version,"
            "memory.total",
            "--format=csv,noheader",
        ]
    )

    return info


# --------------------------------------------------------------- pip freeze

def save_pip_freeze():
    """freeze packages from this exact interpreter."""

    text = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ]
    )

    path = (
        OUT
        / "pip_freeze.txt"
    )

    path.write_text(
        text + "\n",
        encoding="utf-8",
    )

    return path


# --------------------------------------------------------------- hashes

def build_hash_manifest():
    """hash the scripts and result artefacts."""

    rows = []

    missing = []

    for name in ALL_FILES:

        path = Path(name)

        if not path.exists():
            missing.append(
                name
            )
            continue

        stat = path.stat()

        rows.append(
            {
                "path":
                    path.as_posix(),

                "size_bytes":
                    stat.st_size,

                "modified_time":
                    datetime.fromtimestamp(
                        stat.st_mtime,
                        timezone.utc,
                    ).isoformat(),

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    output = (
        OUT
        / "evidence_hashes.csv"
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "size_bytes",
                "modified_time",
                "sha256",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return (
        output,
        rows,
        missing,
    )


# --------------------------------------------------------------- snapshot

def save_environment(info):

    path = (
        OUT
        / "environment_snapshot.json"
    )

    path.write_text(
        json.dumps(
            info,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


# --------------------------------------------------------------- human summary

def write_summary(
    env,
    hash_rows,
    missing,
):
    """make a short human-readable snapshot."""

    path = (
        OUT
        / "REPRODUCIBILITY_SNAPSHOT.md"
    )

    lines = [
        "# Reproducibility snapshot",
        "",
        "This snapshot records the environment and "
        "evidence state at the end of the current "
        "dissertation experiment cycle.",
        "",
        "**Important:** this is a contemporaneous "
        "snapshot of the environment at freeze time. "
        "It does not retroactively prove that every "
        "earlier experiment used this exact environment.",
        "",
        "## Environment",
        "",
        f"- UTC snapshot: `{env['snapshot_utc']}`",
        f"- Python executable: `{env['python_executable']}`",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{env['platform']}`",
        "",
        "### Main package versions",
        "",
    ]

    for name, value in (
        env["packages"].items()
    ):
        lines.append(
            f"- {name}: `{value}`"
        )

    lines.extend(
        [
            "",
            "### CUDA / GPU",
            "",
            f"- CUDA available: "
            f"`{env.get('torch', {}).get('cuda_available')}`",

            f"- PyTorch CUDA: "
            f"`{env.get('torch', {}).get('cuda_version')}`",

            f"- nvidia-smi: "
            f"`{env.get('nvidia_smi')}`",

            "",
            "## Evidence freeze",
            "",
            f"- Files hashed: **{len(hash_rows)}**",
            "- Hash algorithm: **SHA-256**",
            "- Manifest: `evidence_hashes.csv`",
            "- Package snapshot: `pip_freeze.txt`",
            "",
        ]
    )

    if missing:

        lines.extend(
            [
                "## Missing requested files",
                "",
                "The following files were listed for "
                "freezing but were not present:",
                "",
            ]
        )

        for item in missing:
            lines.append(
                f"- `{item}`"
            )

    else:
        lines.extend(
            [
                "## Missing requested files",
                "",
                "None.",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "The hashes identify the exact current "
            "scripts, datasets and result artefacts "
            "used for the dissertation evidence freeze.",
            "",
            "If any hashed file changes after this "
            "snapshot, its SHA-256 will no longer match "
            "this manifest.",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


# --------------------------------------------------------------- run

if __name__ == "__main__":

    print(
        "freezing current reproducibility state"
    )

    print(
        "python :",
        sys.executable,
    )

    print(
        "note   : no experiments or model calls"
    )

    print()

    env = collect_environment()

    env_path = save_environment(
        env
    )

    pip_path = save_pip_freeze()

    (
        hash_path,
        hash_rows,
        missing,
    ) = build_hash_manifest()

    summary_path = write_summary(
        env,
        hash_rows,
        missing,
    )

    print(
        "environment:"
    )

    print(
        " python       :",
        platform.python_version(),
    )

    print(
        " torch        :",
        env["packages"]["torch"],
    )

    print(
        " transformers :",
        env["packages"]["transformers"],
    )

    print(
        " tokenizers   :",
        env["packages"]["tokenizers"],
    )

    print(
        " cuda         :",
        env.get(
            "torch",
            {},
        ).get(
            "cuda_available"
        ),
    )

    print()

    print(
        "files hashed :",
        len(hash_rows),
    )

    print(
        "files missing:",
        len(missing),
    )

    if missing:

        for item in missing:
            print(
                "  missing:",
                item,
            )

    print()

    print(
        "outputs:"
    )

    print(
        " ",
        env_path,
    )

    print(
        " ",
        pip_path,
    )

    print(
        " ",
        hash_path,
    )

    print(
        " ",
        summary_path,
    )

    print()

    print(
        "reproducibility freeze complete"
    )