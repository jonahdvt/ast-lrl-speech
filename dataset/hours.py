#!/usr/bin/env python3
"""
Script to calculate the number of training hours in the FLEURS  dataset.
FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) 
is a multilingual speech dataset by Google.
"""

import librosa
import numpy as np
from datasets import load_dataset
def calculate_audio_duration(audio_array, sample_rate):
    """Calculate duration of audio in seconds."""
    return len(audio_array) / sample_rate

def load_and_calculate_hours():
    """
    Load FLEURS dataset and calculate total training hours for .
    
    Returns:
        dict: Dictionary containing duration statistics
    """
    language_code = "pa_in"
    split = "train"
    
    print(f"Loading FLEURS dataset for {language_code}, {split} split...")
    
    try:
        # Load the dataset
        dataset = load_dataset("google/fleurs", language_code, split=split)
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
        
        total_duration_seconds = 0
        durations = []
        
        print("Calculating audio durations...")
        
        # First, let's examine the structure of the first sample
        if len(dataset) > 0:
            first_sample = dataset[0]
            print(f"Sample keys: {list(first_sample.keys())}")
            if 'audio' in first_sample:
                audio_info = first_sample['audio']
                print(f"Audio keys: {list(audio_info.keys()) if isinstance(audio_info, dict) else type(audio_info)}")
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                
                # Get audio data
                audio = sample['audio']
                audio_array = audio['array']
                sample_rate = audio['sampling_rate']
                
                # Calculate duration
                duration = len(audio_array) / sample_rate
                durations.append(duration)
                total_duration_seconds += duration
                
                # Print progress every 100 samples
                if (i + 1) % 100 == 0:
                    current_hours = total_duration_seconds / 3600
                    print(f"Processed {i + 1}/{len(dataset)} samples. Current total: {current_hours:.2f} hours")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Continue with next sample instead of failing completely
                continue
        
        # Calculate statistics
        total_hours = total_duration_seconds / 3600
        total_minutes = total_duration_seconds / 60
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        stats = {
            'total_samples': len(dataset),
            'total_seconds': total_duration_seconds,
            'total_minutes': total_minutes,
            'total_hours': total_hours,
            'avg_duration_seconds': avg_duration,
            'min_duration_seconds': min_duration,
            'max_duration_seconds': max_duration,
            'language_code': language_code,
            'split': split
        }
        
        return stats
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def print_statistics(stats):
    """Print formatted statistics."""
    if stats is None:
        print("No statistics to display.")
        return
    
    print("\n" + "="*50)
    print(f"FLEURS {stats['language_code']} Dataset Statistics ({stats['split']} split)")
    print("="*50)
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Total duration: {stats['total_hours']:.2f} hours")
    print(f"Total duration: {stats['total_minutes']:.2f} minutes")
    print(f"Total duration: {stats['total_seconds']:.2f} seconds")
    print(f"Average sample duration: {stats['avg_duration_seconds']:.2f} seconds")
    print(f"Shortest sample: {stats['min_duration_seconds']:.2f} seconds")
    print(f"Longest sample: {stats['max_duration_seconds']:.2f} seconds")
    print("="*50)

def main():
    """Main function to run the script."""
    print(f"FLEURS Training Hours Calculator")
    print("="*40)
    
    stats = load_and_calculate_hours()
    print_statistics(stats)

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import datasets
        import numpy
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install datasets numpy")
        exit(1)
    
    main()