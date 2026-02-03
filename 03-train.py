#!/usr/bin/env python3
"""
VibeVoice Training Script

A simplified training script for fine-tuning VibeVoice models with sensible defaults.
This script wraps the original finetune_vibevoice_lora with an easier-to-use interface.

Usage:
    # Basic training
    python train.py --dataset data/dataset.jsonl
    
    # With custom output and model
    python train.py --dataset data/dataset.jsonl --output_dir ./my_model --model 1.5B
    
    # Resume from checkpoint
    python train.py --dataset data/dataset.jsonl --resume_from_checkpoint ./my_model/checkpoint-500
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


# Preset configurations for different model sizes
MODEL_PRESETS = {
    "1.5B": {
        "model_name": "aoi-ot/VibeVoice-Base",
        "recommended_vram": "16GB",
        "batch_size": 4,
        "gradient_accumulation": 16,
        "lora_r": 8,
        "lora_alpha": 32,
    },
    "7B": {
        "model_name": "aoi-ot/VibeVoice-Large",
        "recommended_vram": "48GB",
        "batch_size": 2,
        "gradient_accumulation": 32,
        "lora_r": 8,
        "lora_alpha": 32,
    },
    "large": {
        "model_name": "aoi-ot/VibeVoice-Large",
        "recommended_vram": "48GB",
        "batch_size": 2,
        "gradient_accumulation": 32,
        "lora_r": 8,
        "lora_alpha": 32,
    },
    "base": {
        "model_name": "aoi-ot/VibeVoice-Base",
        "recommended_vram": "16GB",
        "batch_size": 4,
        "gradient_accumulation": 16,
        "lora_r": 8,
        "lora_alpha": 32,
    },
}

# Preset configurations for different training modes
TRAINING_PRESETS = {
    "fast": {
        "description": "Quick training for testing",
        "num_epochs": 1,
        "learning_rate": 5e-5,
        "save_steps": 50,
        "eval_steps": 50,
        "logging_steps": 5,
    },
    "default": {
        "description": "Balanced quality and speed",
        "num_epochs": 5,
        "learning_rate": 2.5e-5,
        "save_steps": 100,
        "eval_steps": 100,
        "logging_steps": 10,
    },
    "quality": {
        "description": "Best quality, slower training",
        "num_epochs": 10,
        "learning_rate": 1e-5,
        "save_steps": 200,
        "eval_steps": 200,
        "logging_steps": 10,
    },
}


def get_project_root() -> Path:
    """Get the project root directory."""
    # Look for VibeVoice-finetuning directory
    current = Path.cwd()
    
    # Check current directory
    if (current / "VibeVoice-finetuning" / "src").exists():
        return current / "VibeVoice-finetuning"
    
    # Check if we're inside the repo
    if (current / "src" / "finetune_vibevoice_lora.py").exists():
        return current
    
    # Check parent directories
    for parent in current.parents:
        if (parent / "VibeVoice-finetuning" / "src").exists():
            return parent / "VibeVoice-finetuning"
        if (parent / "src" / "finetune_vibevoice_lora.py").exists():
            return parent
    
    return current


def check_environment() -> bool:
    """Check if the environment is properly set up."""
    project_root = get_project_root()
    
    # Check if VibeVoice-finetuning exists
    if not (project_root / "src" / "finetune_vibevoice_lora.py").exists():
        print("Error: VibeVoice-finetuning not found!")
        print("Please run setup.sh first:")
        print("  bash setup.sh")
        return False
    
    # Check if virtual environment is active
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: Virtual environment not detected!")
        print("It's recommended to activate the environment first:")
        print("  source activate_env.sh")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True


def check_model_downloaded(model_name: str) -> Optional[Path]:
    """Check if model is already downloaded locally."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        return None
    
    # Check for model directory
    model_dir_name = model_name.replace("/", "--")
    model_path = models_dir / model_dir_name
    
    if model_path.exists():
        return model_path
    
    return None


def build_training_command(args) -> List[str]:
    """Build the training command with all arguments."""
    project_root = get_project_root()
    
    # Get preset configurations
    model_preset = MODEL_PRESETS.get(args.model, MODEL_PRESETS["1.5B"])
    training_preset = TRAINING_PRESETS.get(args.preset, TRAINING_PRESETS["default"])
    
    # Determine model path
    model_path = args.model_path
    if not model_path:
        # Check if model is downloaded locally
        local_model = check_model_downloaded(model_preset["model_name"])
        if local_model:
            model_path = str(local_model)
        else:
            model_path = model_preset["model_name"]
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.finetune_vibevoice_lora",
        "--model_name_or_path", model_path,
        "--processor_name_or_path", "src/vibevoice/processor",
        "--output_dir", args.output_dir,
        "--text_column_name", "text",
        "--audio_column_name", "audio",
        "--remove_unused_columns", "False",
        "--bf16", "True",
        "--do_train",
    ]
    
    # Dataset
    if args.val_dataset:
        cmd.extend(["--train_jsonl", args.dataset])
        cmd.extend(["--validation_jsonl", args.val_dataset])
    else:
        cmd.extend(["--train_jsonl", args.dataset])
    
    # Training hyperparameters (use args if provided, otherwise use presets)
    batch_size = args.batch_size if args.batch_size else model_preset["batch_size"]
    grad_accum = args.gradient_accumulation if args.gradient_accumulation else model_preset["gradient_accumulation"]
    num_epochs = args.num_epochs if args.num_epochs else training_preset["num_epochs"]
    learning_rate = args.learning_rate if args.learning_rate else training_preset["learning_rate"]
    
    cmd.extend(["--per_device_train_batch_size", str(batch_size)])
    cmd.extend(["--gradient_accumulation_steps", str(grad_accum)])
    cmd.extend(["--num_train_epochs", str(num_epochs)])
    cmd.extend(["--learning_rate", str(learning_rate)])
    
    # LoRA configuration
    lora_r = args.lora_r if args.lora_r else model_preset["lora_r"]
    lora_alpha = args.lora_alpha if args.lora_alpha else model_preset["lora_alpha"]
    
    cmd.extend(["--lora_r", str(lora_r)])
    cmd.extend(["--lora_alpha", str(lora_alpha)])
    cmd.extend(["--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"])
    
    # Logging and saving
    save_steps = args.save_steps if args.save_steps else training_preset["save_steps"]
    eval_steps = args.eval_steps if args.eval_steps else training_preset["eval_steps"]
    logging_steps = args.logging_steps if args.logging_steps else training_preset["logging_steps"]
    
    cmd.extend(["--save_steps", str(save_steps)])
    cmd.extend(["--eval_steps", str(eval_steps)])
    cmd.extend(["--logging_steps", str(logging_steps)])
    
    # Reporting
    if args.wandb:
        cmd.extend(["--report_to", "wandb"])
    elif args.tensorboard:
        cmd.extend(["--report_to", "tensorboard"])
    else:
        cmd.extend(["--report_to", "none"])
    
    # Optimizer settings
    cmd.extend(["--lr_scheduler_type", "cosine"])
    cmd.extend(["--warmup_ratio", "0.03"])
    cmd.extend(["--max_grad_norm", "0.8"])
    
    # Gradient clipping
    cmd.append("--gradient_clipping")
    
    # Memory optimization
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    
    # Diffusion training settings
    cmd.extend(["--ddpm_batch_mul", str(args.ddpm_batch_mul)])
    cmd.extend(["--diffusion_loss_weight", str(args.diffusion_loss_weight)])
    cmd.extend(["--ce_loss_weight", str(args.ce_loss_weight)])
    
    if args.train_diffusion_head:
        cmd.append("--train_diffusion_head")
    
    if args.train_connectors:
        cmd.append("--train_connectors")
    
    # Voice prompt settings
    if args.voice_prompt_column:
        cmd.extend(["--voice_prompts_column_name", args.voice_prompt_column])
        cmd.extend(["--voice_prompt_drop_rate", str(args.voice_prompt_drop_rate)])
    
    # Resume from checkpoint
    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    
    # Custom arguments
    for arg in args.extra_args:
        cmd.append(arg)
    
    return cmd


def print_training_config(args):
    """Print a summary of the training configuration."""
    model_preset = MODEL_PRESETS.get(args.model, MODEL_PRESETS["1.5B"])
    training_preset = TRAINING_PRESETS.get(args.preset, TRAINING_PRESETS["default"])
    
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model: {model_preset['model_name']} ({args.model})")
    print(f"Recommended VRAM: {model_preset['recommended_vram']}")
    print(f"Dataset: {args.dataset}")
    if args.val_dataset:
        print(f"Validation: {args.val_dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nPreset: {args.preset} - {training_preset['description']}")
    print(f"  Epochs: {args.num_epochs or training_preset['num_epochs']}")
    print(f"  Learning rate: {args.learning_rate or training_preset['learning_rate']}")
    print(f"  Batch size: {args.batch_size or model_preset['batch_size']}")
    print(f"  Gradient accumulation: {args.gradient_accumulation or model_preset['gradient_accumulation']}")
    print(f"\nLoRA config:")
    print(f"  r: {args.lora_r or model_preset['lora_r']}")
    print(f"  alpha: {args.lora_alpha or model_preset['lora_alpha']}")
    print(f"\nDiffusion training:")
    print(f"  DDPM batch multiplier: {args.ddpm_batch_mul}")
    print(f"  Diffusion loss weight: {args.diffusion_loss_weight}")
    print(f"  CE loss weight: {args.ce_loss_weight}")
    print(f"  Train diffusion head: {args.train_diffusion_head}")
    print("=" * 60 + "\n")


def estimate_training_time(args) -> str:
    """Estimate training time based on configuration."""
    # This is a rough estimate
    model_preset = MODEL_PRESETS.get(args.model, MODEL_PRESETS["1.5B"])
    training_preset = TRAINING_PRESETS.get(args.preset, TRAINING_PRESETS["default"])
    
    # Count dataset entries
    try:
        with open(args.dataset) as f:
            num_samples = sum(1 for _ in f)
    except:
        num_samples = 1000  # Default assumption
    
    effective_batch = (args.batch_size or model_preset["batch_size"]) * \
                     (args.gradient_accumulation or model_preset["gradient_accumulation"])
    steps_per_epoch = num_samples // effective_batch
    total_steps = steps_per_epoch * (args.num_epochs or training_preset["num_epochs"])
    
    # Rough time estimate (very approximate)
    time_per_step = 2.0 if args.model in ["7B", "large"] else 1.0
    total_hours = (total_steps * time_per_step) / 3600
    
    if total_hours < 1:
        return f"~{int(total_hours * 60)} minutes"
    else:
        return f"~{total_hours:.1f} hours"


def main():
    parser = argparse.ArgumentParser(
        description='Train VibeVoice models with LoRA fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python train.py --dataset data/dataset.jsonl --preset fast
  
  # Standard training with 1.5B model
  python train.py --dataset data/dataset.jsonl --model 1.5B
  
  # High quality training with 7B model
  python train.py --dataset data/dataset.jsonl --model 7B --preset quality
  
  # Resume from checkpoint
  python train.py --dataset data/dataset.jsonl --resume_from_checkpoint ./output/checkpoint-500
  
  # Custom hyperparameters
  python train.py --dataset data/dataset.jsonl --num_epochs 3 --learning_rate 1e-5 --batch_size 2
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', '--train_dataset', type=str, required=True,
                        help='Path to training dataset JSONL file')
    
    # Model selection
    parser.add_argument('--model', type=str, default='1.5B',
                        choices=list(MODEL_PRESETS.keys()),
                        help='Model size to train (default: 1.5B)')
    parser.add_argument('--model_path', type=str,
                        help='Direct path to model (overrides --model)')
    
    # Dataset options
    parser.add_argument('--val_dataset', '--validation_jsonl', type=str,
                        help='Path to validation dataset JSONL file')
    parser.add_argument('--voice_prompt_column', type=str,
                        help='Column name for voice prompts in dataset')
    parser.add_argument('--voice_prompt_drop_rate', type=float, default=0.2,
                        help='Voice prompt dropout rate (default: 0.2)')
    
    # Training preset
    parser.add_argument('--preset', type=str, default='default',
                        choices=list(TRAINING_PRESETS.keys()),
                        help='Training preset (default: default)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for model checkpoints (default: output)')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int,
                        help='Number of training epochs (overrides preset)')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate (overrides preset)')
    parser.add_argument('--batch_size', '--per_device_train_batch_size', type=int,
                        help='Batch size per device (overrides preset)')
    parser.add_argument('--gradient_accumulation', '--gradient_accumulation_steps', type=int,
                        help='Gradient accumulation steps (overrides preset)')
    
    # LoRA configuration
    parser.add_argument('--lora_r', type=int,
                        help='LoRA rank (default: from preset)')
    parser.add_argument('--lora_alpha', type=int,
                        help='LoRA alpha (default: from preset)')
    
    # Logging and saving
    parser.add_argument('--save_steps', type=int,
                        help='Save checkpoint every N steps (overrides preset)')
    parser.add_argument('--eval_steps', type=int,
                        help='Evaluate every N steps (overrides preset)')
    parser.add_argument('--logging_steps', type=int,
                        help='Log every N steps (overrides preset)')
    
    # Reporting
    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument('--wandb', action='store_true',
                              help='Enable Weights & Biases logging')
    report_group.add_argument('--tensorboard', action='store_true',
                              help='Enable TensorBoard logging')
    
    # Memory optimization
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    
    # Diffusion training
    parser.add_argument('--ddpm_batch_mul', type=int, default=4,
                        help='DDPM batch multiplier (default: 4)')
    parser.add_argument('--diffusion_loss_weight', type=float, default=1.4,
                        help='Diffusion loss weight (default: 1.4)')
    parser.add_argument('--ce_loss_weight', type=float, default=0.04,
                        help='Cross-entropy loss weight (default: 0.04)')
    parser.add_argument('--train_diffusion_head', action='store_true', default=True,
                        help='Train diffusion head (default: True)')
    parser.add_argument('--train_connectors', action='store_true',
                        help='Train acoustic and semantic connectors')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str,
                        help='Resume from checkpoint directory')
    
    # Pass-through arguments
    parser.add_argument('extra_args', nargs='*',
                        help='Additional arguments to pass to the training script')
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Print configuration
    print_training_config(args)
    
    # Estimate training time
    estimated_time = estimate_training_time(args)
    print(f"Estimated training time: {estimated_time}")
    print(f"(This is a rough estimate and depends on your hardware)\n")
    
    # Confirm before starting
    if not args.resume_from_checkpoint:
        response = input("Start training? (Y/n): ")
        if response.lower() == 'n':
            print("Training cancelled.")
            sys.exit(0)
    
    # Build and run command
    cmd = build_training_command(args)
    project_root = get_project_root()
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    print("Command:")
    print(" \\\n  ".join(cmd))
    print()
    
    # Run training
    try:
        result = subprocess.run(cmd, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("To resume, use:")
        print(f"  python train.py --dataset {args.dataset} --resume_from_checkpoint {args.output_dir}/checkpoint-XXX")
        sys.exit(1)


if __name__ == '__main__':
    main()
