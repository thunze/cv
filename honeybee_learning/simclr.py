# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.


# Imports
import torch
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from torch import nn

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader, Dataset


class SimCLR(nn.Module):
    """
        SimCLR model adapted for the honeybee problem task
    """
    def __init__(self, backbone, hidden_dim):
        super().__init__()
        self.backbone = backbone

        # Input and hidden dim of same size in paper
        self.projection_head = SimCLRProjectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


# Initialize hyperparams
epochs = 100 
batch_size = 4096 # cf. Paper Figure 9 (for small epoch size large batch sizes perform signficantly better)
lr = 0.3*batch_size/256 # as defined in paper
output_dim = 128 # for projection head

device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers=8

# Model initialization
resnet = torchvision.models.resnet50()
backbone = nn.Sequential(*list(resnet.children())[:-1])
hidden_dim = resnet.fc.in_features # equals to 2048 for ResNet50
model = SimCLR(backbone, hidden_dim)

model.to(device)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs) # optional but highly recommended in paper (especially with lr defined as is)


def train_simclr():
    '''
        Training pipeline for SimCLR model
    '''

    # TODO: run wandb
    
    # Load training and validation data
    train_dataset = load_dataset(mode="train")
    validate_dataset = load_dataset(mode="validate")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False, # Turned off for validation data
        drop_last=True,
        num_workers=num_workers,
    )

    
    ## Training ## 
    print("Starting SimCLR Training")
    model.train()
    model.zero_grad()
    for epoch in range(epochs):
        training_loss = 0
        for batch in train_dataloader:
            x0, x1 = batch[0] # NOTE: dependent on getitem format
            
            x0 = x0.to(device)
            x1 = x1.to(device)
            
            z0 = model(x0)
            z1 = model(x1)
            
            loss = criterion(z0, z1)
            training_loss += loss.detach()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        avg_training_loss = training_loss / len(train_dataloader)

        
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in validate_dataloader:
                x0, x1 = batch[0] # NOTE: dependent on getitem format
                
                x0 = x0.to(device)
                x1 = x1.to(device)
                
                z0 = model(x0)
                z1 = model(x1)
                
                loss = criterion(z0, z1)
                validation_loss += loss.detach()

        # Adapt learning rate (optional)
        scheduler.step()
    
        avg_validation_loss = validation_loss / len(validate_dataloader)
    
        print(f"Epoch: {epoch:>02}, AVG Train Loss: {avg_training_loss:.5f}, AVG Val Loss: {avg_validation_loss:.5f}")
        
    