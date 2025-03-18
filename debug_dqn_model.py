import torch
import os
import argparse
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def debug_model(model_path):
    
    print(f"Debugging DQN model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist at {model_path}")
        return
    
    file_size = os.path.getsize(model_path)
    print(f"Model file size: {file_size} bytes ({file_size / (1024 * 1024):.2f} MB)")
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("\nCheckpoint loaded successfully!")
        
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"- {key}")
        
        # Determine if model parameters are stored directly or under 'model_state_dict'
        has_model_state_dict = 'model_state_dict' in checkpoint
        has_direct_params = any(key.endswith('.weight') or key.endswith('.bias') for key in checkpoint.keys())
        
        # Process model parameters
        if has_model_state_dict:
            state_dict = checkpoint['model_state_dict']
            print("\nModel state dict structure:")
        elif has_direct_params:
            state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
            print("\nModel parameters structure (stored directly in checkpoint):")
        else:
            print("\nNo model parameters found in checkpoint")
            state_dict = {}
        
        if state_dict:
            # Group parameters by component
            q_network_params = 0
            feature_extractor_params = 0
            other_params = 0
            
            # Count total trainable parameters
            total_trainable_params = 0
            
            for key, value in state_dict.items():
                print(f"- {key}: Shape {value.shape}, Parameters: {value.numel()}")
                
                # Count parameters by component
                if 'features' in key or 'conv' in key:
                    feature_extractor_params += value.numel()
                elif 'fc' in key or 'linear' in key or 'q_net' in key:
                    q_network_params += value.numel()
                else:
                    other_params += value.numel()
                
                # Count trainable parameters (exclude running stats)
                if not any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                    total_trainable_params += value.numel()
            
            # Print parameter counts by component
            total_params = q_network_params + feature_extractor_params + other_params
            print("\nParameter counts by component:")
            print(f"- Q-Network: {q_network_params:,} parameters ({q_network_params/total_params*100:.2f}%)")
            if feature_extractor_params > 0:
                print(f"- Feature extractor: {feature_extractor_params:,} parameters ({feature_extractor_params/total_params*100:.2f}%)")
            if other_params > 0:
                print(f"- Other: {other_params:,} parameters ({other_params/total_params*100:.2f}%)")
            print(f"- Total: {total_params:,} parameters")
            
            # Print trainable parameters count
            print(f"\nTrainable parameters (excluding running stats): {total_trainable_params:,}")
            
            # If the model itself is included in the checkpoint, use count_parameters function
            if 'model' in checkpoint:
                model_trainable_params = count_parameters(checkpoint['model'])
                print(f"Trainable parameters (using count_parameters): {model_trainable_params:,}")
        
        # If target_network_state_dict exists, check if it matches the model state dict
        if 'target_network_state_dict' in checkpoint:
            print("\nTarget network state dict found!")
            print("Checking if target network matches Q-network...")
            
            if has_model_state_dict:
                # Compare shapes of parameters
                matches = True
                for (k1, v1), (k2, v2) in zip(
                    checkpoint['model_state_dict'].items(),
                    checkpoint['target_network_state_dict'].items()
                ):
                    if v1.shape != v2.shape:
                        print(f"Shape mismatch: {k1} {v1.shape} vs {k2} {v2.shape}")
                        matches = False
                    
                    # Check if values are identical (would indicate target network hasn't been updated)
                    if torch.all(v1 == v2):
                        print(f"Warning: {k1} has identical values in both networks")
                
                if matches:
                    print("Target network has matching architecture with Q-network")
                    
                # Calculate average difference between networks
                total_diff = 0
                total_params = 0
                for (k1, v1), (k2, v2) in zip(
                    checkpoint['model_state_dict'].items(),
                    checkpoint['target_network_state_dict'].items()
                ):
                    if v1.shape == v2.shape:
                        diff = torch.abs(v1 - v2).mean().item()
                        total_diff += diff * v1.numel()
                        total_params += v1.numel()
                
                if total_params > 0:
                    avg_diff = total_diff / total_params
                    print(f"Average parameter difference between Q-network and target network: {avg_diff:.6f}")
        
        # If optimizer_state_dict exists, print its structure
        if 'optimizer_state_dict' in checkpoint:
            print("\nOptimizer state dict structure:")
            print(f"- param_groups: {len(checkpoint['optimizer_state_dict']['param_groups'])} groups")
            for i, group in enumerate(checkpoint['optimizer_state_dict']['param_groups']):
                print(f"  - Group {i}: {len(group['params'])} parameters, lr={group['lr']}")
        
        # Check for DQN-specific hyperparameters and metadata
        print("\nDQN-specific metadata:")
        dqn_params = ['epsilon', 'gamma', 'target_update_freq', 'batch_size', 'buffer_size', 
                      'learning_rate', 'episode', 'total_steps', 'rewards']
        
        for param in dqn_params:
            if param in checkpoint:
                print(f"- {param}: {checkpoint[param]}")
        
        # Print other metadata if available
        print("\nOther metadata:")
        for key in checkpoint.keys():
            if key not in ['model_state_dict', 'target_network_state_dict', 'optimizer_state_dict'] and key not in dqn_params:
                # Skip parameters already printed in model structure
                if has_direct_params and isinstance(checkpoint[key], torch.Tensor):
                    continue
                
                # Handle special cases for certain types of data
                if isinstance(checkpoint[key], (int, float, str, bool)):
                    print(f"- {key}: {checkpoint[key]}")
                elif isinstance(checkpoint[key], (list, np.ndarray)) and len(checkpoint[key]) < 10:
                    print(f"- {key}: {checkpoint[key]}")
                elif isinstance(checkpoint[key], (list, np.ndarray)):
                    print(f"- {key}: Array of length {len(checkpoint[key])}")
                else:
                    print(f"- {key}: {type(checkpoint[key])}")
        
        # If replay buffer is saved, analyze it
        if 'replay_buffer' in checkpoint:
            print("\nReplay buffer information:")
            buffer = checkpoint['replay_buffer']
            if hasattr(buffer, 'capacity'):
                print(f"- Capacity: {buffer.capacity}")
            if hasattr(buffer, 'size') or hasattr(buffer, 'position'):
                size = getattr(buffer, 'size', getattr(buffer, 'position', None))
                print(f"- Current size: {size}")
                print(f"- Fill percentage: {size/buffer.capacity*100:.2f}%")
        
    except Exception as e:
        print(f"Error loading model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Debug a saved DQN model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model')
    
    args = parser.parse_args()
    
    debug_model(args.model_path)

if __name__ == "__main__":
    main() 