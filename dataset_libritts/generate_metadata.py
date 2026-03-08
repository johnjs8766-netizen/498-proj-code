import os
import pandas as pd
import random

def create_libritts_metadata(dataset_root, output_csv="dataset/metadata.csv"):
    data = []
    # Supported extensions
    valid_extensions = ('.wav', '.flac')
    
    print(f"Crawling {dataset_root}...")
    
    # Walk through Speaker/Chapter folders
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(valid_extensions):
                full_path = os.path.join(root, file)
                # Assign a random target device label (0-5)
                # This ensures all generators see all types of speech
                device_label = random.randint(0, 5)
                
                data.append({
                    "filename": full_path,
                    "device": device_label
                })

    df = pd.DataFrame(data)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f"Done! Created {output_csv} with {len(df)} entries.")

if __name__ == "__main__":
    # Change this to the actual path of your LibriTTS folder
    # Example: "dataset/LibriTTS/train-clean-100"
    LIBRITTS_PATH = "./" 
    create_libritts_metadata(LIBRITTS_PATH)