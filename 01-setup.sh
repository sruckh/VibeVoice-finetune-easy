#!/bin/bash
# VibeVoice Fine-tuning Setup Script
# This script sets up the complete environment for VibeVoice fine-tuning

set -e  # Exit on error

echo "====================================="
echo "VibeVoice Fine-tuning Setup"
echo "====================================="

# Configuration
PYTHON_VERSION="3.11"
VENV_DIR="venv"
PROJECT_DIR="$(pwd)"
MODELS_DIR="${PROJECT_DIR}/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on CUDA-enabled system
check_cuda() {
    print_status "Checking CUDA availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        print_status "CUDA is available!"
    else
        print_warning "nvidia-smi not found. Make sure you have NVIDIA drivers installed."
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-venv \
            git \
            git-lfs \
            ffmpeg \
            libsndfile1 \
            sox \
            libsox-dev
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        sudo yum install -y \
            python3-pip \
            git \
            git-lfs \
            ffmpeg \
            libsndfile \
            sox \
            sox-devel
    elif command -v brew &> /dev/null; then
        # macOS
        brew install \
            python@3.11 \
            git \
            git-lfs \
            ffmpeg \
            libsndfile \
            sox
    else
        print_warning "Could not detect package manager. Please install dependencies manually."
    fi
    
    # Initialize git-lfs
    git lfs install || true
}

# Create Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "${VENV_DIR}" ]; then
        print_warning "Virtual environment already exists. Removing old environment..."
        rm -rf "${VENV_DIR}"
    fi
    
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Pin numpy to <2.0.0 for numba compatibility
    pip install "numpy<2.0.0"
    
    print_status "Virtual environment created at ${VENV_DIR}"
}

# Clone the VibeVoice-finetuning repository
clone_repository() {
    if [ ! -d "VibeVoice-finetuning" ]; then
        print_status "Cloning VibeVoice-finetuning repository..."
        git clone https://github.com/voicepowered-ai/VibeVoice-finetuning.git
    else
        print_status "Repository already exists, pulling latest changes..."
        cd VibeVoice-finetuning
        git pull
        cd ..
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    source "${VENV_DIR}/bin/activate"
    
    # Install PyTorch with CUDA support
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install VibeVoice-finetuning package
    print_status "Installing VibeVoice-finetuning package..."
    cd VibeVoice-finetuning
    pip install -e .
    
    # Install specific transformers version that works with VibeVoice
    print_status "Installing compatible transformers version..."
    pip uninstall -y transformers
    pip install transformers==4.51.3
    
    # Install additional utilities
    print_status "Installing additional utilities..."
    pip install \
        datasets \
        soundfile \
        librosa \
        accelerate \
        bitsandbytes \
        wandb \
        tensorboard \
        pydub \
        audioread
    
    cd ..
}

# Download models
download_models() {
    print_status "Setting up models directory..."
    mkdir -p "${MODELS_DIR}"
    
    source "${VENV_DIR}/bin/activate"
    
    print_status "Downloading VibeVoice model files..."
    print_warning "This may take a while depending on your internet connection."
    
    # Create a Python script to download models
    python3 << 'EOF'
import os
from huggingface_hub import snapshot_download

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Model options
models = {
    "1": {
        "name": "VibeVoice-Large (7B)",
        "repo": "aoi-ot/VibeVoice-Large",
        "desc": "Best quality, requires 48GB VRAM for training"
    },
    "2": {
        "name": "VibeVoice-Base (1.5B)",
        "repo": "aoi-ot/VibeVoice-Base",
        "desc": "Faster training, requires 16GB VRAM for training"
    }
}

print("\nAvailable models:")
for key, model in models.items():
    print(f"  {key}. {model['name']} - {model['desc']}")

choice = input("\nWhich model would you like to download? (1/2/both): ").strip().lower()

to_download = []
if choice == "1":
    to_download.append(models["1"])
elif choice == "2":
    to_download.append(models["2"])
elif choice in ["both", "all", "b"]:
    to_download = list(models.values())
else:
    print("Invalid choice. Downloading VibeVoice-Base (1.5B) as default.")
    to_download.append(models["2"])

for model in to_download:
    print(f"\nDownloading {model['name']}...")
    try:
        snapshot_download(
            repo_id=model["repo"],
            local_dir=os.path.join(models_dir, model["repo"].replace("/", "--")),
            local_dir_use_symlinks=False
        )
        print(f"✓ {model['name']} downloaded successfully!")
    except Exception as e:
        print(f"✗ Error downloading {model['name']}: {e}")
        print("  You can download it later manually.")

print("\nModel download complete!")
EOF
}

# Create helper scripts
create_helper_scripts() {
    print_status "Creating helper scripts..."
    
    # Create activate script
    cat > activate_env.sh << EOF
#!/bin/bash
source "${PROJECT_DIR}/${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/VibeVoice-finetuning:\$PYTHONPATH"
export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${PROJECT_DIR}/hf_cache/datasets"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/hf_cache/transformers"
echo "VibeVoice environment activated!"
echo "Python: \$(which python)"
echo ""
echo "Available commands:"
echo "  python prepare_dataset.py  - Prepare your dataset"
echo "  python train.py            - Start training"
EOF
    chmod +x activate_env.sh
    
    # Create cache directories
    mkdir -p hf_cache/datasets
    mkdir -p hf_cache/transformers
    mkdir -p data/audio
    mkdir -p data/output
}

# Print summary
print_summary() {
    echo ""
    echo "====================================="
    echo "Setup Complete!"
    echo "====================================="
    echo ""
    echo "Project directory: ${PROJECT_DIR}"
    echo "Virtual environment: ${VENV_DIR}"
    echo "Models directory: ${MODELS_DIR}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the environment:"
    echo "   source activate_env.sh"
    echo ""
    echo "2. Prepare your dataset:"
    echo "   python prepare_dataset.py --audio_dir /path/to/your/audio --output data/dataset.jsonl"
    echo ""
    echo "3. Start training:"
    echo "   python train.py --dataset data/dataset.jsonl"
    echo ""
    echo "For help, run:"
    echo "   python prepare_dataset.py --help"
    echo "   python train.py --help"
    echo ""
}

# Main execution
main() {
    print_status "Starting VibeVoice fine-tuning setup..."
    
    check_cuda
    install_system_deps
    setup_venv
    clone_repository
    install_python_deps
    download_models
    create_helper_scripts
    
    print_summary
}

# Run main function
main "$@"
