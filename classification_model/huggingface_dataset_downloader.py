# 1. Imports and Setup
import os
import pandas as pd
from datasets import load_dataset
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Dataset Loading Functions
def load_misinformation_dataset(split="train", max_samples=None):
    logger.info(f"Loading misinformation dataset (split={split})...")
    
    try:
        dataset = load_dataset(
            "daviddaubner/misinformation-detection",
            split=split
        )
        
        df = dataset.to_pandas()
        
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            
        logger.info(f"Successfully loaded {len(df)} samples from misinformation dataset")
        return df
        
    except Exception as e:
        logger.error(f"Error loading misinformation dataset: {str(e)}")
        return None

def load_covid_fake_news_dataset(split="train", max_samples=None):
    logger.info(f"Loading COVID fake news dataset (split={split})...")
    
    try:
        dataset = load_dataset(
            "nanyy1025/covid_fake_news",
            split=split
        )
        
        df = dataset.to_pandas()
        
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            
        logger.info(f"Successfully loaded {len(df)} samples from COVID fake news dataset")
        return df
        
    except Exception as e:
        logger.error(f"Error loading COVID fake news dataset: {str(e)}")
        return None

def load_pubhealth_dataset(split="train", max_samples=None):
    logger.info(f"Loading PubHealth dataset (split={split})...")
    
    try:
        dataset = load_dataset(
            "health_fact",
            split=split
        )
        
        df = dataset.to_pandas()
        
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            
        logger.info(f"Successfully loaded {len(df)} samples from PubHealth dataset")
        return df
        
    except Exception as e:
        logger.error(f"Error loading PubHealth dataset: {str(e)}")
        return None

def load_scifact_dataset(split="train", max_samples=None):
    logger.info(f"Loading SciFact dataset (split={split})...")
    
    try:
        dataset = load_dataset(
            "scifact",
            split=split
        )
        
        df = dataset.to_pandas()
        
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            
        logger.info(f"Successfully loaded {len(df)} samples from SciFact dataset")
        return df
        
    except Exception as e:
        logger.error(f"Error loading SciFact dataset: {str(e)}")
        return None

# 3. Dataset Download Function
def download_datasets(output_dir="data", max_samples=5000, datasets=None):
    if datasets is None:
        datasets = ["misinformation", "covid", "pubhealth", "scifact"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_files = []
    
    if "misinformation" in datasets:
        for split in ["train", "validation", "test"]:
            try:
                logger.info(f"Downloading misinformation dataset ({split} split)...")
                df = load_misinformation_dataset(split=split, max_samples=max_samples)
                
                if df is not None:
                    output_file = os.path.join(output_dir, f"misinformation_{split}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved {len(df)} samples to {output_file}")
                    downloaded_files.append(output_file)
            except Exception as e:
                logger.error(f"Error downloading misinformation dataset ({split}): {str(e)}")
    
    if "covid" in datasets:
        for split in ["train", "validation", "test"]:
            try:
                logger.info(f"Downloading COVID fake news dataset ({split} split)...")
                df = load_covid_fake_news_dataset(split=split, max_samples=max_samples)
                
                if df is not None:
                    output_file = os.path.join(output_dir, f"covid_fake_news_{split}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved {len(df)} samples to {output_file}")
                    downloaded_files.append(output_file)
            except Exception as e:
                logger.error(f"Error downloading COVID fake news dataset ({split}): {str(e)}")
    
    if "pubhealth" in datasets:
        for split in ["train", "validation", "test"]:
            try:
                logger.info(f"Downloading PubHealth dataset ({split} split)...")
                df = load_pubhealth_dataset(split=split, max_samples=max_samples)
                
                if df is not None:
                    output_file = os.path.join(output_dir, f"pubhealth_{split}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved {len(df)} samples to {output_file}")
                    downloaded_files.append(output_file)
            except Exception as e:
                logger.error(f"Error downloading PubHealth dataset ({split}): {str(e)}")
    
    if "scifact" in datasets:
        for split in ["train", "validation", "test"]:
            try:
                logger.info(f"Downloading SciFact dataset ({split} split)...")
                df = load_scifact_dataset(split=split, max_samples=max_samples)
                
                if df is not None:
                    output_file = os.path.join(output_dir, f"scifact_{split}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved {len(df)} samples to {output_file}")
                    downloaded_files.append(output_file)
            except Exception as e:
                logger.error(f"Error downloading SciFact dataset ({split}): {str(e)}")
    
    logger.info(f"Downloaded {len(downloaded_files)} dataset files to {output_dir}")
    return downloaded_files

# 4. Command Line Interface
def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the downloaded datasets"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of samples to download per dataset split"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["misinformation", "covid", "pubhealth", "scifact", "all"],
        default=["all"],
        help="Datasets to download"
    )
    
    return parser.parse_args()

# 5. Main Function
def main():
    args = parse_args()
    
    if "all" in args.datasets:
        datasets = ["misinformation", "covid", "pubhealth", "scifact"]
    else:
        datasets = args.datasets
    
    logger.info(f"Downloading datasets: {', '.join(datasets)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max samples per split: {args.max_samples}")
    
    download_datasets(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        datasets=datasets
    )
    
    logger.info("Download complete!")

if __name__ == "__main__":
    main()