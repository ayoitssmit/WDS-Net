import torch
import torch.nn as nn

class WDSNet(nn.Module):
    def __init__(self, num_classes):
        super(WDSNet, self).__init__()
        
        # Path 1: Spatial Feature Learning (Deep CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: (16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: (32, 7, 7)
        )
        
        # Path 2: Structural Dependency Learning (LSTM)
        # Using feature maps from CNN: 32 channels, treating 7 rows as seq steps, 32*7 features/step.
        self.lstm = nn.LSTM(input_size=32*7, hidden_size=128, batch_first=True)
        
        # Path 3: Global Feature Learning
        # Extracted externally: Length = 1 (mean) + 1 (variance) + 16 (histogram bins) = 18 elements
        self.global_dim = 18
        
        # Fusion & Classification
        # Spatial vector flat size = 32 * 7 * 7 = 1568
        # Structural vector = 128 (hidden size)
        # Global vector = 18
        fused_dim = 1568 + 128 + self.global_dim
        
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            # Note: Softmax is implicitly applied during training by nn.CrossEntropyLoss
        )

    def forward(self, x, global_features):
        batch_size = x.size(0)
        device = x.device
        
        # 1. Spatial Path Execution
        cnn_out = self.cnn(x)
        spatial_features = cnn_out.view(batch_size, -1)
        
        # 2. Structural Path Execution
        # Reshaping cnn_out into sequence: (Batch, Sequence, Feature_Dim) -> (Batch, 7, 32*7)
        lstm_input = cnn_out.view(batch_size, 32, 7, 7).permute(0, 2, 1, 3).contiguous()
        lstm_input = lstm_input.view(batch_size, 7, 32*7)
        
        # Explicit initialization of h0 and c0 on the CORRECT device
        h0 = torch.zeros(1, batch_size, 128).to(device)
        c0 = torch.zeros(1, batch_size, 128).to(device)
        
        _, (h_n, _) = self.lstm(lstm_input, (h0, c0))
        structural_features = h_n[-1] # Grabbing last hidden state
        
        # 3. Global Path execution is already precomputed in global_features parameter passed in
        
        # 4. Feature Fusion (Concatenation)
        fused_features = torch.cat((spatial_features, structural_features, global_features), dim=1)
        
        # 5. Classification
        logits = self.fc(fused_features)
        
        return logits
