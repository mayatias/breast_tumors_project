import torch
from training_pipeline import run_experiment
from model import CResUNet

lr = 5e-5
epochs = 20
num_workers = 2
batch_size = 8
n_splits = 1
root_dir = "BUSI_Dataset"
test_ratio = 0.2
seed = 42

def print_model_parameter_count():
    model = CResUNet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: total={total_params:,} | trainable={trainable_params:,}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print_model_parameter_count()
    run_experiment(root_dir=root_dir, n_splits=n_splits, batch_size=batch_size, epochs=epochs,
                   lr=lr, test_ratio=test_ratio, seed=seed, device=device, num_workers=num_workers)

if __name__ == "__main__":
    main()