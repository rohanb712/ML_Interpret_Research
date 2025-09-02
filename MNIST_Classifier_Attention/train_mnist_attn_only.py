import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model, num_heads = 192, 4
        
        # Patch embedding: 4x4 patches -> 192 dims
        self.patch_embed = nn.Linear(16, d_model)
        
        # Learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model))  # 49 patches + 1 CLS
        
        # Single attention block
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
        # Classifier
        self.head = nn.Linear(d_model, 10)
        
    def forward(self, x):
        # Convert image to patches: [B, 1, 28, 28] -> [B, 49, 16]
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
        
        # Embed patches
        x = self.patch_embed(x)  # [B, 49, 192]
        
        # Add CLS token and positional encoding
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.size(0))
        x = torch.cat([cls_token, x], dim=1)  # [B, 50, 192]
        x = x + self.pos_embed
        
        # Self-attention
        x_norm = self.norm(x)
        qkv = rearrange(self.qkv(x_norm), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (48 ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention and project
        out = rearrange(attn @ v, 'b h n d -> b n (h d)')
        out = self.proj(out)
        x = x + out  # Residual connection
        
        # Classification from CLS token
        return self.head(x[:, 0])

def main():
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)
    ])
    
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)
    
    # Model and training
    model = AttentionTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train for 5 epochs
    for epoch in range(5):
        model.train()
        train_loss, train_correct = 0, 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += output.argmax(1).eq(target).sum().item()
        
        # Test
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_correct += output.argmax(1).eq(target).sum().item()
        
        train_acc = 100 * train_correct / len(train_data)
        test_acc = 100 * test_correct / len(test_data)
        
        print(f"Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%")

if __name__ == "__main__":
    main()