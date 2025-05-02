import pandas as pd

# Your dataset path
dataset_path = "tsl_dataset.csv"

# The columns your backend expects
required_columns = ['Home', 'Away', 'HomeGoals', 'AwayGoals']
optional_columns = ['HomeRedCards', 'AwayRedCards']

def validate_dataset_structure(path):
    try:
        df = pd.read_csv(path)
        columns = df.columns.tolist()
        
        print(f"\n📄 Columns in CSV: {columns}\n")

        missing_required = [col for col in required_columns if col not in columns]
        missing_optional = [col for col in optional_columns if col not in columns]
        
        if missing_required:
            print(f"❌ Missing REQUIRED columns: {missing_required}")
        else:
            print("✅ All required columns are present.")

        if missing_optional:
            print(f"⚠️ Missing OPTIONAL columns (they will be filled with 0 in backend): {missing_optional}")
        else:
            print("✅ All optional columns are present.")

    except Exception as e:
        print(f"❌ Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    validate_dataset_structure(dataset_path)