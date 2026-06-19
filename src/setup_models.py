import os
import sys

def setup():
    print("="*60)
    print("STARTING REDROB OFFLINE MODELS SETUP")
    print("="*60)
    
    # 1. Download and save the Cross-Encoder ranking model
    print("\n[1/2] Downloading Cross-Encoder model ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
    try:
        from sentence_transformers import CrossEncoder
        local_dir = "models/cross-encoder"
        
        # This downloads and loads the model (cached by huggingface)
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Save it locally in the repository
        print(f"Saving Cross-Encoder model locally to '{local_dir}'...")
        os.makedirs(local_dir, exist_ok=True)
        model.save(local_dir)
        print("--> Cross-Encoder model saved successfully!")
    except Exception as e:
        print(f"Error setting up Cross-Encoder model: {e}")
        sys.exit(1)
        
    # 2. Download the Qwen-0.5B GGUF model
    print("\n[2/2] Downloading Qwen2.5-0.5B GGUF model from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        
        gguf_filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        
        local_dir = "models/qwen-2.5"
        print(f"Downloading '{gguf_filename}' to '{local_dir}'...")
        hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename=gguf_filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"--> Qwen GGUF model saved successfully at '{local_dir}/{gguf_filename}'!")
    except Exception as e:
        print(f"Error setting up Qwen GGUF model: {e}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("SETUP COMPLETE! All models saved locally for offline sandbox run.")
    print("="*60 + "\n")

if __name__ == "__main__":
    setup()
