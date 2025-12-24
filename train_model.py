"""
Model Training Script
This script trains all classification models on the resume dataset
Supports both automatic download and manual dataset paths
"""

import os


import sys
import pandas as pd
from pathlib import Path
from resume_classifier_utils import ResumeClassifier


def get_dataset_path():
    """Get dataset path from user or command line argument"""
    # Check if path provided as command-line argument
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        if Path(dataset_path).exists():
            print(f"✅ Dataset path provided: {dataset_path}")
            
            return dataset_path
        else:
            print(f"❌ Path not found: {dataset_path}")
            sys.exit(1)

    # Prompt user for manual path
    print("\n📂 DATASET LOCATION")
    print("="*60)
    print("Enter the path to your manually downloaded dataset folder")
    print("(It should contain the 'UpdatedResumeDataSet.csv' file)")
    print("\nExample paths:")
    print("  • /home/user/Downloads/resume-dataset")
    print("  • C:\\Users\\user\\Downloads\\resume-dataset")
    print("  • ./data/resume-dataset")
    print("-"*60)

    dataset_path = input("Dataset folder path: ").strip()

    if not dataset_path:
        print("\n❌ No path provided!")
        print("\nAlternative: Provide path as command-line argument:")
        print("  python train_model.py /path/to/dataset")
        sys.exit(1)

    dataset_path = Path(dataset_path).expanduser().resolve()

    if not dataset_path.exists():
        print(f"❌ Path not found: {dataset_path}")
        sys.exit(1)

    print(f"✅ Using dataset path: {dataset_path}")
    return str(dataset_path)


def load_csv_from_path(dataset_path):
    """Load CSV file from the downloaded dataset path"""
    print("\n📂 Loading CSV file...")
    files = os.listdir(dataset_path)
    csv_files = [f for f in files if f.endswith('.csv')]

    if not csv_files:
        print(f"❌ No CSV files found in {dataset_path}")
        sys.exit(1)

    csv_file = csv_files[0]
    csv_path = os.path.join(dataset_path, csv_file)
    print(f"✅ Loading: {csv_file}")

    df = pd.read_csv(csv_path)
    print(f"✅ Dataset loaded: {df.shape[0]} resumes, {df.shape[1]} columns")
    return df


def train_all_models(df):
    """Train all classification models"""
    models = ['logistic', 'naive_bayes', 'knn']
    results = {}

    print("\n" + "="*60)
    print("🤖 TRAINING CLASSIFICATION MODELS")
    print("="*60)

    for model_type in models:
        print(f"\n\n📌 Training {model_type.upper().replace('_', ' ')} Model...")
        print("-" * 60)

        classifier = ResumeClassifier(model_type=model_type)

        try:
            # Train the model
            training_info = classifier.train(
                X=df['Resume'],
                y=df['Category'],
                test_size=0.2,
                random_state=42
            )

            print(f"✅ Training completed!")
            print(f"   • Accuracy: {training_info['accuracy']:.4f} ({training_info['accuracy']*100:.2f}%)")
            print(f"   • Train set size: {training_info['X_train_shape'][0]} samples")
            print(f"   • Test set size: {training_info['X_test_shape'][0]} samples")

            # Save the model
            print(f"\n💾 Saving {model_type.upper().replace('_', ' ')} model...")
            classifier.save_model('models')
            print(f"   ✅ Model saved to: models/{model_type}_model.pkl")

            results[model_type] = {
                'accuracy': training_info['accuracy'],
                'report': training_info['report']
            }

        except Exception as e:
            print(f"❌ Error training {model_type}: {str(e)}")
            raise

    return results


def print_summary(results):
    """Print training summary"""
    print("\n\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)

    for model_name, result in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0].upper().replace('_', ' ')} "
          f"({best_model[1]['accuracy']*100:.2f}% accuracy)")

    print("\n" + "="*60)
    print("🎉 All models trained successfully!")
    print("="*60)
    print("\n▶️  Next step: Run 'streamlit run app.py' to start the application")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🚀 RESUME CLASSIFIER - MODEL TRAINING")
    print("="*60)

    # Check if models directory exists
    Path('models').mkdir(exist_ok=True)

    # Step 1: Get dataset path
    dataset_path = get_dataset_path()

    # Step 2: Load data
    df = load_csv_from_path(dataset_path)

    # Display dataset info
    print("\n📊 Dataset Information:")
    print(f"   • Total resumes: {len(df)}")
    print(f"   • Job categories: {df['Category'].nunique()}")
    print(f"   • Categories: {', '.join(sorted(df['Category'].unique()[:5]))}...")

    # Step 3: Train models
    results = train_all_models(df)

    # Step 4: Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
