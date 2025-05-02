"""
Dataset Analysis Examples

This script demonstrates how to load and analyze the datasets 
downloaded using the huggingface_dataset_downloader.py script.
"""

# 1. Imports and Setup
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from wordcloud import WordCloud



try:
    from datasets import load_dataset
except ImportError:
    print("Hugging Face datasets library not installed. You won't be able to load datasets directly from Hugging Face.")
    print("Install it with: pip install datasets")

plt.style.use('ggplot')
DATA_DIR = "data"

# 2. Data Loading Functions
def load_dataset(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    return pd.read_csv(filepath)

def load_misinformation_dataset(split="train", max_samples=None):
    print(f"Loading misinformation dataset (split={split}) from Hugging Face...")
    
    try:
        dataset = load_dataset(
            "daviddaubner/misinformation-detection",
            split=split
        )
        
        df = dataset.to_pandas()
        
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            
        print(f"Successfully loaded {len(df)} samples from misinformation dataset")
        return df
        
    except Exception as e:
        print(f"Error loading misinformation dataset: {str(e)}")
        return None

def load_covid_fake_news_dataset(split="train", max_samples=None):
    print(f"Loading COVID fake news dataset (split={split}) from Hugging Face...")
    
    try:
        dataset = load_dataset(
            "nanyy1025/covid_fake_news",
            split=split
        )
        
        df = dataset.to_pandas()
        
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            
        print(f"Successfully loaded {len(df)} samples from COVID fake news dataset")
        return df
        
    except Exception as e:
        print(f"Error loading COVID fake news dataset: {str(e)}")
        return None

# 3. Dataset Analysis Functions
def analyze_pubhealth(df):
    print("\n" + "="*50)
    print("ANALYZING PUBHEALTH DATASET")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"- {label}: {count} ({count/len(df)*100:.1f}%)")
            
        plt.figure(figsize=(10, 6))
        sns.countplot(y='label', data=df)
        plt.title('Distribution of Labels in PubHealth Dataset')
        plt.tight_layout()
        plt.savefig('pubhealth_labels.png')
        print("Label distribution chart saved as 'pubhealth_labels.png'")
    
    if 'main_text' in df.columns:
        df['text_length'] = df['main_text'].apply(len)
        
        print("\nText length statistics:")
        print(f"- Average length: {df['text_length'].mean():.1f} characters")
        print(f"- Median length: {df['text_length'].median():.1f} characters")
        print(f"- Min length: {df['text_length'].min()} characters")
        print(f"- Max length: {df['text_length'].max()} characters")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'], bins=30)
        plt.title('Distribution of Text Lengths in PubHealth Dataset')
        plt.xlabel('Text Length (characters)')
        plt.tight_layout()
        plt.savefig('pubhealth_text_length.png')
        print("Text length distribution chart saved as 'pubhealth_text_length.png'")
    
    return df

def analyze_health_misinfo(df):
    print("\n" + "="*50)
    print("ANALYZING HEALTH_MISINFO DATASET")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    label_columns = [col for col in df.columns if 'label' in col.lower()]
    for label_col in label_columns:
        label_counts = df[label_col].value_counts()
        print(f"\n{label_col} distribution:")
        for label, count in label_counts.items():
            print(f"- {label}: {count} ({count/len(df)*100:.1f}%)")
    
    text_columns = [col for col in df.columns if any(text_type in col.lower() 
                                                    for text_type in ['text', 'content', 'claim'])]
    
    if text_columns:
        main_text_col = text_columns[0] 
        df['text_length'] = df[main_text_col].apply(len)
        
        print(f"\n{main_text_col} length statistics:")
        print(f"- Average length: {df['text_length'].mean():.1f} characters")
        print(f"- Median length: {df['text_length'].median():.1f} characters")
        print(f"- Min length: {df['text_length'].min()} characters")
        print(f"- Max length: {df['text_length'].max()} characters")
    
    return df

def analyze_scifact(df):
    print("\n" + "="*50)
    print("ANALYZING SCIFACT DATASET")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    if 'claim' in df.columns:
        df['claim_length'] = df['claim'].apply(len)
        
        print("\nClaim length statistics:")
        print(f"- Average length: {df['claim_length'].mean():.1f} characters")
        print(f"- Median length: {df['claim_length'].median():.1f} characters")
        print(f"- Min length: {df['claim_length'].min()} characters")
        print(f"- Max length: {df['claim_length'].max()} characters")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df['claim_length'], bins=30)
        plt.title('Distribution of Claim Lengths in SciFact Dataset')
        plt.xlabel('Claim Length (characters)')
        plt.tight_layout()
        plt.savefig('scifact_claim_length.png')
        print("Claim length distribution chart saved as 'scifact_claim_length.png'")
    
    evidence_col = next((col for col in df.columns if 'evidence' in col.lower()), None)
    if evidence_col:
        print(f"\nSample evidence from first record: {df[evidence_col].iloc[0]}")
    
    return df

def analyze_misinformation(df):
    print("\n" + "="*50)
    print("ANALYZING MISINFORMATION DETECTION DATASET")
    print("="*50)
    
    if df is None or len(df) == 0:
        print("No data available for analysis.")
        return None
    
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    text_col = None
    for col in ['text', 'content', 'claim', 'sentence', 'statement']:
        if col in df.columns:
            text_col = col
            break
    
    label_col = None
    for col in ['label', 'class', 'is_fake', 'fake', 'misinformation', 'is_misinformation']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        label_counts = df[label_col].value_counts()
        print(f"\n{label_col} distribution:")
        for label, count in label_counts.items():
            print(f"- {label}: {count} ({count/len(df)*100:.1f}%)")
        
        plt.figure(figsize=(10, 6))
        sns.countplot(y=label_col, data=df)
        plt.title('Distribution of Labels in Misinformation Dataset')
        plt.tight_layout()
        plt.savefig('misinformation_labels.png')
        print("Label distribution chart saved as 'misinformation_labels.png'")
    
    if text_col:
        df['text_length'] = df[text_col].apply(len)
        df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
        
        print(f"\n{text_col} length statistics:")
        print(f"- Average length: {df['text_length'].mean():.1f} characters")
        print(f"- Median length: {df['text_length'].median():.1f} characters")
        print(f"- Min length: {df['text_length'].min()} characters")
        print(f"- Max length: {df['text_length'].max()} characters")
        
        print(f"\nWord count statistics:")
        print(f"- Average word count: {df['word_count'].mean():.1f} words")
        print(f"- Median word count: {df['word_count'].median():.1f} words")
        print(f"- Min word count: {df['word_count'].min()} words")
        print(f"- Max word count: {df['word_count'].max()} words")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'], bins=30)
        plt.title('Distribution of Text Lengths in Misinformation Dataset')
        plt.xlabel('Text Length (characters)')
        plt.tight_layout()
        plt.savefig('misinformation_text_length.png')
        print("Text length distribution chart saved as 'misinformation_text_length.png'")
        
        try:
            plt.figure(figsize=(12, 8))
            text = ' '.join(df[text_col].astype(str))
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100
            ).generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('misinformation_wordcloud.png')
            print("Word cloud visualization saved as 'misinformation_wordcloud.png'")
        except Exception as e:
            print(f"Error generating word cloud: {e}")
        
        if label_col:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=label_col, y='text_length', data=df)
            plt.title('Text Length by Label in Misinformation Dataset')
            plt.tight_layout()
            plt.savefig('misinformation_length_by_label.png')
            print("Text length by label chart saved as 'misinformation_length_by_label.png'")
    
    return df

def analyze_covid_fake_news(df):
    print("\n" + "="*50)
    print("ANALYZING COVID-19 FAKE NEWS DATASET")
    print("="*50)
    
    if df is None or len(df) == 0:
        print("No data available for analysis.")
        return None
    
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    text_col = None
    for col in ['text', 'content', 'title', 'news', 'claim']:
        if col in df.columns:
            text_col = col
            break
    
    label_col = None
    for col in ['label', 'class', 'is_fake', 'fake']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        label_counts = df[label_col].value_counts()
        print(f"\n{label_col} distribution:")
        for label, count in label_counts.items():
            print(f"- {label}: {count} ({count/len(df)*100:.1f}%)")
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(y=label_col, data=df)
        plt.title('Distribution of Labels in COVID-19 Fake News Dataset')
        plt.tight_layout()
        plt.savefig('covid_fake_news_labels.png')
        print("Label distribution chart saved as 'covid_fake_news_labels.png'")
    
    if text_col:
        df['text_length'] = df[text_col].apply(len)
        df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
        
        print(f"\n{text_col} length statistics:")
        print(f"- Average length: {df['text_length'].mean():.1f} characters")
        print(f"- Median length: {df['text_length'].median():.1f} characters")
        print(f"- Min length: {df['text_length'].min()} characters")
        print(f"- Max length: {df['text_length'].max()} characters")
        
        print(f"\nWord count statistics:")
        print(f"- Average word count: {df['word_count'].mean():.1f} words")
        print(f"- Median word count: {df['word_count'].median():.1f} words")
        print(f"- Min word count: {df['word_count'].min()} words")
        print(f"- Max word count: {df['word_count'].max()} words")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'], bins=30)
        plt.title('Distribution of Text Lengths in COVID-19 Fake News Dataset')
        plt.xlabel('Text Length (characters)')
        plt.tight_layout()
        plt.savefig('covid_fake_news_text_length.png')
        print("Text length distribution chart saved as 'covid_fake_news_text_length.png'")
        
        covid_terms = ['covid', 'coronavirus', 'pandemic', 'vaccine', 'virus', 
                      'mask', 'quarantine', 'lockdown', 'social distancing']
        
        term_counts = {}
        for term in covid_terms:
            count = df[text_col].str.lower().str.contains(term).sum()
            term_counts[term] = count
        
        print("\nCOVID-related term frequencies:")
        for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {term}: {count} ({count/len(df)*100:.1f}%)")
        
        plt.figure(figsize=(12, 8))
        term_df = pd.DataFrame(list(term_counts.items()), columns=['Term', 'Count'])
        term_df = term_df.sort_values('Count', ascending=False)
        
        sns.barplot(x='Count', y='Term', data=term_df)
        plt.title('COVID-Related Term Frequencies')
        plt.tight_layout()
        plt.savefig('covid_term_frequencies.png')
        print("COVID term frequencies chart saved as 'covid_term_frequencies.png'")
        
        try:
            plt.figure(figsize=(12, 8))
            text = ' '.join(df[text_col].astype(str))
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100
            ).generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('covid_fake_news_wordcloud.png')
            print("Word cloud visualization saved as 'covid_fake_news_wordcloud.png'")
        except Exception as e:
            print(f"Error generating word cloud: {e}")
    
    return df

def compare_datasets(datasets):
    print("\n" + "="*50)
    print("COMPARING DATASETS")
    print("="*50)
    
    print("Dataset sizes:")
    for name, df in datasets.items():
        if df is not None:
            print(f"- {name}: {len(df)} records")

def analyze_model_info(model_name="mav23_Medichat-Llama3-8B-GGUF"):
    print("\n" + "="*50)
    print(f"ANALYZING {model_name} MODEL INFO")
    print("="*50)
    
    filepath = os.path.join(DATA_DIR, f"{model_name}_info.json")
    if not os.path.exists(filepath):
        print(f"Model info file not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        model_info = json.load(f)
    
    print("Model Details:")
    print(f"- ID: {model_info.get('id', 'N/A')}")
    print(f"- Pipeline Tag: {model_info.get('pipeline_tag', 'N/A')}")
    
    if 'tags' in model_info:
        print("\nTags:")
        for tag in model_info['tags']:
            print(f"- {tag}")
    
    if 'siblings' in model_info:
        print("\nModel Files:")
        for filename in model_info['siblings'][:10]:  
            print(f"- {filename}")
        
        if len(model_info['siblings']) > 10:
            print(f"... and {len(model_info['siblings']) - 10} more files")

def download_huggingface_datasets(
    output_dir=DATA_DIR,
    max_samples=5000,
    datasets_to_download=["misinformation", "covid"]
):
    print("\n" + "="*50)
    print("DOWNLOADING DATASETS FROM HUGGING FACE")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_files = []
    
    if "misinformation" in datasets_to_download:
        for split in ["train", "validation", "test"]:
            try:
                print(f"Downloading misinformation dataset ({split} split)...")
                df = load_misinformation_dataset(split=split, max_samples=max_samples)
                
                if df is not None:
                    output_file = os.path.join(output_dir, f"misinformation_{split}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"Saved {len(df)} samples to {output_file}")
                    downloaded_files.append(output_file)
            except Exception as e:
                print(f"Error downloading misinformation dataset ({split}): {str(e)}")
    
    if "covid" in datasets_to_download:
        for split in ["train", "validation", "test"]:
            try:
                print(f"Downloading COVID fake news dataset ({split} split)...")
                df = load_covid_fake_news_dataset(split=split, max_samples=max_samples)
                
                if df is not None:
                    output_file = os.path.join(output_dir, f"covid_fake_news_{split}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"Saved {len(df)} samples to {output_file}")
                    downloaded_files.append(output_file)
            except Exception as e:
                print(f"Error downloading COVID fake news dataset ({split}): {str(e)}")
    
    print(f"\nDownloaded {len(downloaded_files)} dataset files to {output_dir}")
    return downloaded_files

def main():
    print("Starting dataset analysis...")
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found! Creating it now.")
        os.makedirs(DATA_DIR)
    
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the data directory. Downloading datasets from Hugging Face...")
        try:
            downloaded_files = download_huggingface_datasets(DATA_DIR)
            
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        except Exception as e:
            print(f"Error downloading datasets: {str(e)}")
            print("You can install the required libraries with: pip install datasets")
    
    print(f"Found {len(csv_files)} CSV files in the data directory:")
    for filename in csv_files:
        print(f"- {filename}")
    
    datasets = {}
    
    pubhealth_file = next((f for f in csv_files if 'pubhealth' in f.lower()), None)
    if pubhealth_file:
        df_pubhealth = load_dataset(pubhealth_file)
        if df_pubhealth is not None:
            datasets['pubhealth'] = analyze_pubhealth(df_pubhealth)
    
    health_misinfo_file = next((f for f in csv_files if 'health_misinfo' in f.lower()), None)
    if health_misinfo_file:
        df_health_misinfo = load_dataset(health_misinfo_file)
        if df_health_misinfo is not None:
            datasets['health_misinfo'] = analyze_health_misinfo(df_health_misinfo)
    
    scifact_file = next((f for f in csv_files if 'scifact' in f.lower()), None)
    if scifact_file:
        df_scifact = load_dataset(scifact_file)
        if df_scifact is not None:
            datasets['scifact'] = analyze_scifact(df_scifact)
            
    misinfo_file = next((f for f in csv_files if 'misinformation_train' in f.lower()), None)
    if misinfo_file:
        df_misinfo = load_dataset(misinfo_file)
        if df_misinfo is not None:
            datasets['misinfo_detection'] = analyze_misinformation(df_misinfo)
            
    covid_file = next((f for f in csv_files if 'covid_fake_news_train' in f.lower()), None)
    if covid_file:
        df_covid = load_dataset(covid_file)
        if df_covid is not None:
            datasets['covid_fake_news'] = analyze_covid_fake_news(df_covid)
    
    if 'misinfo_detection' not in datasets or 'covid_fake_news' not in datasets:
        try:
            print("\nAttempting to load datasets directly from Hugging Face...")
            
            if 'misinfo_detection' not in datasets:
                df_misinfo = load_misinformation_dataset(max_samples=1000)
                if df_misinfo is not None:
                    datasets['misinfo_detection'] = analyze_misinformation(df_misinfo)
            
            if 'covid_fake_news' not in datasets:
                df_covid = load_covid_fake_news_dataset(max_samples=1000)
                if df_covid is not None:
                    datasets['covid_fake_news'] = analyze_covid_fake_news(df_covid)
                    
        except Exception as e:
            print(f"Error loading datasets directly from Hugging Face: {str(e)}")
    
    if len(datasets) > 1:
        compare_datasets(datasets)
    
    print("\nDataset analysis complete!")

if __name__ == "__main__":
    main()