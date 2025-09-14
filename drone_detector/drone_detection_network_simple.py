import torch
import torch.nn as nn
import torch.nn.functional as F


class DroneDetectionNetwork(nn.Module):
    """
    Simplified neural network for binary drone detection based on acoustic features
    Optimized for small datasets to prevent overfitting
    
    Input: Configurable number of acoustic features (default: 21 from BandFeatureExtractor)
    Output: Single probability value (0-1) indicating drone presence

    Architecture: input_size -> 32 -> 16 -> 1
    """
    
    def __init__(self, 
                 input_size: int = 21,
                 dropout_rate: float = 0.3,
                 leaky_relu_slope: float = 0.01):
        """
        Initialize simplified network with higher regularization for small datasets
        
        Args:
            input_size: Number of input features (default: 21 from BandFeatureExtractor)
            dropout_rate: Dropout probability for regularization (increased for small datasets)
            leaky_relu_slope: Negative slope for LeakyReLU activation
        """
        super(DroneDetectionNetwork, self).__init__()
        
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.leaky_relu_slope = leaky_relu_slope
        
        # Feature scaler for input normalization
        self.feature_scaler = nn.Linear(input_size, input_size, bias=False)
        
        # Layer 1: Input -> 32 (reduced from 128)
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 32 -> 16 (reduced from 64)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer: 16 -> 1 (simplified architecture)
        self.fc_out = nn.Linear(16, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Kaiming initialization for LeakyReLU"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for LeakyReLU activations
                nn.init.kaiming_normal_(module.weight, a=self.leaky_relu_slope, mode='fan_in')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize feature scaler as identity matrix (no scaling initially)
        nn.init.eye_(self.feature_scaler.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the simplified network
        
        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, 1)
        """
        # Feature scaling (learnable input normalization)
        x = self.feature_scaler(x)
        
        # Layer 1: 21 -> 32 (with dropout for regularization)
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=self.leaky_relu_slope)
        x1 = self.dropout1(x1)
        
        # Layer 2: 32 -> 16 (with dropout for regularization)
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=self.leaky_relu_slope)
        x2 = self.dropout2(x2)
        
        # Output layer: 16 -> 1
        output = self.fc_out(x2)
        output = torch.sigmoid(output)
        
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities (inference mode)
        
        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Probabilities of drone presence
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            threshold: Decision threshold for binary classification
            
        Returns:
            torch.Tensor: Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).float()
    
    def get_layer_outputs(self, x: torch.Tensor, layer: str = "fc2") -> torch.Tensor:
        """
        Get outputs from a specific layer (useful for feature analysis)
        
        Args:
            x: Input tensor
            layer: Layer name ("fc1", "fc2", "fc_out")
            
        Returns:
            torch.Tensor: Layer outputs
        """
        self.eval()
        with torch.no_grad():
            # Feature scaling
            x = self.feature_scaler(x)
            
            # Layer 1
            x1 = self.fc1(x)
            x1 = self.bn1(x1)
            x1 = F.leaky_relu(x1, negative_slope=self.leaky_relu_slope)
            if layer == "fc1":
                return x1
            x1 = self.dropout1(x1)
            
            # Layer 2
            x2 = self.fc2(x1)
            x2 = self.bn2(x2)
            x2 = F.leaky_relu(x2, negative_slope=self.leaky_relu_slope)
            if layer == "fc2":
                return x2
            x2 = self.dropout2(x2)
            
            # Output layer
            output = self.fc_out(x2)
            return torch.sigmoid(output)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get detailed network architecture summary"""
        summary_str = f"DroneDetectionNetwork Summary:\n"
        summary_str += f"{'='*50}\n"
        summary_str += f"Input size: {self.input_size}\n"
        summary_str += f"Architecture: {self.input_size} -> 32 -> 16 -> 1\n"
        summary_str += f"Dropout rate: {self.dropout_rate}\n"
        summary_str += f"LeakyReLU slope: {self.leaky_relu_slope}\n"
        summary_str += f"Total parameters: {self.count_parameters():,}\n"
        summary_str += f"{'='*50}\n"
        
        # Layer-wise parameter count
        summary_str += "Layer-wise details:\n"
        summary_str += f"feature_scaler:    {self.feature_scaler.weight.numel():,} params (no bias)\n"
        summary_str += f"fc1 ({self.input_size}->32):      {self.fc1.weight.numel() + self.fc1.bias.numel():,} params\n"
        summary_str += f"bn1:               {self.bn1.weight.numel() + self.bn1.bias.numel():,} params\n"
        summary_str += f"fc2 (32->16):      {self.fc2.weight.numel() + self.fc2.bias.numel():,} params\n"
        summary_str += f"bn2:               {self.bn2.weight.numel() + self.bn2.bias.numel():,} params\n"
        summary_str += f"fc_out (16->1):    {self.fc_out.weight.numel() + self.fc_out.bias.numel():,} params\n"
        
        return summary_str
    
    def get_architecture_info(self) -> dict:
        """Get architecture information as dictionary"""
        return {
            'input_size': self.input_size,
            'architecture': f"{self.input_size} -> 32 -> 16 -> 1",
            'dropout_rate': self.dropout_rate,
            'leaky_relu_slope': self.leaky_relu_slope,
            'total_parameters': self.count_parameters(),
            'num_layers': 3  # 2 hidden + 1 output
        }


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance
    Useful for small datasets with potential class imbalance
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class (typically between 0.25 and 1.0)
            gamma: Focusing parameter (typically between 1.0 and 5.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predictions (after sigmoid)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Example usage and testing
if __name__ == "__main__":
    print("Testing Simplified DroneDetectionNetwork...")

    # Create network with higher dropout for small datasets
    network = DroneDetectionNetwork(
        input_size=21,
        dropout_rate=0.3,
        leaky_relu_slope=0.01  
    )
    
    print(network.summary())
    print()
    
    # Test forward pass
    batch_size = 16
    test_input = torch.randn(batch_size, 21)  # Using default input size
    
    # Training mode
    network.train()
    train_output = network(test_input)
    print(f"Training mode output shape: {train_output.shape}")
    print(f"Training mode output range: [{train_output.min():.3f}, {train_output.max():.3f}]")
    print()
    
    # Inference mode
    probs = network.predict_proba(test_input)
    predictions = network.predict(test_input, threshold=0.5)
    
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Predictions (threshold=0.5): {predictions.sum().item()}/{batch_size} positive")
    print()
    
    # Test different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        preds = network.predict(test_input, threshold=threshold)
        print(f"Threshold {threshold}: {preds.sum().item()}/{batch_size} positive predictions")
    
    print()
    print("Architecture info:")
    arch_info = network.get_architecture_info()
    for key, value in arch_info.items():
        print(f"  {key}: {value}")
    
    # Test layer outputs
    print("\nTesting layer outputs:")
    for layer in ["fc1", "fc2", "fc_out"]:
        output = network.get_layer_outputs(test_input, layer=layer)
        print(f"  {layer} output shape: {output.shape}")
    
    # Test Focal Loss
    print("\nTesting Focal Loss:")
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    dummy_targets = torch.randint(0, 2, (batch_size, 1)).float()
    loss_value = focal_loss(probs, dummy_targets)
    print(f"  Focal loss value: {loss_value.item():.4f}")
    
    print("\nNetwork tested successfully!")