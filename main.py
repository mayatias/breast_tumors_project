import torch
from training_pipeline import run_experiment

lr = 3e-4
epochs = 3
batch_size = 4
n_splits = 5
root_dir = "BUSI_Dataset"
test_ratio = 0.2
seed = 42

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    run_experiment(root_dir=root_dir, n_splits=n_splits, batch_size=batch_size, epochs=epochs,
                   lr=lr, test_ratio=test_ratio, seed=seed, device=device)

if __name__ == "__main__":
    main()