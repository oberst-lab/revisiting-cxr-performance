#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import sys

def clean_csv(input_file, output_file):
    """
    Clean CSV file by keeping only specific columns and removing duplicates.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the cleaned CSV file
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Keep only the required columns
    required_columns = ['label_name', 'classifier_name', 'AUROC']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Select only the required columns
    df_cleaned = df[required_columns].copy()
    
    print(f"After column selection: {df_cleaned.shape}")
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    print(f"After removing duplicates: {df_cleaned.shape}")
    
    # Display some statistics
    print(f"\nNumber of unique labels: {df_cleaned['label_name'].nunique()}")
    print(f"Number of unique classifiers: {df_cleaned['classifier_name'].nunique()}")
    
    # Show duplicates that were removed (if any)
    original_count = len(df[required_columns])
    cleaned_count = len(df_cleaned)
    duplicates_removed = original_count - cleaned_count
    
    if duplicates_removed > 0:
        print(f"\nRemoved {duplicates_removed} duplicate rows")
    else:
        print("\nNo duplicate rows found")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Save the cleaned data
    try:
        df_cleaned.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        sys.exit(1)
    
    # Display first few rows of cleaned data
    print("\nFirst few rows of cleaned data:")
    print(df_cleaned.head())
    
    return df_cleaned

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Clean CSV file by keeping only label_name, classifier_name, and AUROC columns, and removing duplicates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_result_csv.py -i input.csv -o output.csv
  python clean_result_csv.py --input all_results.csv --output cleaned_results.csv
  python clean_result_csv.py -i input.csv -o output.csv --no-stats
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        '-o', '--output', 
        type=str,
        required=True,
        help='Path to save the cleaned CSV file'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Skip printing detailed statistics (default: False)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    # Clean the CSV
    cleaned_df = clean_csv(args.input, args.output)
    
    # Show summary statistics unless --no-stats is specified
    if cleaned_df is not None and not args.no_stats:
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        # Group by label to see classifiers per label
        label_counts = cleaned_df.groupby('label_name').size().sort_values(ascending=False)
        print(f"\nNumber of classifiers per label:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} classifiers")
        
        # Group by classifier to see labels per classifier
        classifier_counts = cleaned_df.groupby('classifier_name').size().sort_values(ascending=False)
        print(f"\nNumber of labels per classifier:")
        for classifier, count in classifier_counts.items():
            print(f"  {classifier}: {count} labels")
        
        # Show AUROC statistics
        print(f"\nAUROC Statistics:")
        print(f"  Mean AUROC: {cleaned_df['AUROC'].mean():.4f}")
        print(f"  Min AUROC: {cleaned_df['AUROC'].min():.4f}")
        print(f"  Max AUROC: {cleaned_df['AUROC'].max():.4f}")
        print(f"  Std AUROC: {cleaned_df['AUROC'].std():.4f}")
    
    print(f"\nProcessing completed successfully!")

if __name__ == "__main__":
    main()