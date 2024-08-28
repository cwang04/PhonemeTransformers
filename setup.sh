if [ ! -d "env" ]; then
	module load python/3.8
    virtualenv -p python3.8 env
	source env/bin/activate
	pip install -r requirements.txt
	install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pre-commit install
	huggingface-cli login
	wandb login
else 
	source env/bin/activate
fi
source .env
export PATH="$(pwd)/lib/bin:$PATH"
git lfs install

