import torch
import os
import argparse

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def debug_model(model_path):
    print(f"Debugging model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist at {model_path}")
        return
    
    file_size = os.path.getsize(model_path)
    print(f"Model file size: {file_size} bytes ({file_size / (1024 * 1024):.2f} MB)")
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("\nCheckpoint loaded successfully!")
        
        # Print the keys in the checkpoint
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"- {key}")
        
        # If model_state_dict exists, print its structure and count parameters
        if 'model_state_dict' in checkpoint:
            print("\nModel state dict structure:")
            
            # Group parameters by component
            actor_params = 0
            critic_params = 0
            feature_params = 0
            other_params = 0
            
            for key, value in checkpoint['model_state_dict'].items():
                print(f"- {key}: Shape {value.shape}, Parameters: {value.numel()}")
                
                # Count parameters by component
                if 'actor' in key:
                    actor_params += value.numel()
                elif 'critic' in key:
                    critic_params += value.numel()
                elif 'features' in key:
                    feature_params += value.numel()
                else:
                    other_params += value.numel()
            
            # Print parameter counts by component
            total_params = actor_params + critic_params + feature_params + other_params
            print("\nParameter counts by component:")
            print(f"- Actor network: {actor_params:,} parameters ({actor_params/total_params*100:.2f}%)")
            print(f"- Critic network: {critic_params:,} parameters ({critic_params/total_params*100:.2f}%)")
            if feature_params > 0:
                print(f"- Feature extractor: {feature_params:,} parameters ({feature_params/total_params*100:.2f}%)")
            if other_params > 0:
                print(f"- Other: {other_params:,} parameters ({other_params/total_params*100:.2f}%)")
            print(f"- Total: {total_params:,} parameters")
        
        # If optimizer_state_dict exists, print its structure
        if 'optimizer_state_dict' in checkpoint:
            print("\nOptimizer state dict structure:")
            print(f"- param_groups: {len(checkpoint['optimizer_state_dict']['param_groups'])} groups")
            for i, group in enumerate(checkpoint['optimizer_state_dict']['param_groups']):
                print(f"  - Group {i}: {len(group['params'])} parameters, lr={group['lr']}")
        
        print("\nOther metadata:")
        for key in checkpoint.keys():
            if key not in ['model_state_dict', 'optimizer_state_dict']:
                print(f"- {key}: {checkpoint[key]}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Debug a saved PPO model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model')
    
    args = parser.parse_args()
    
    debug_model(args.model_path)

if __name__ == "__main__":
    main() 