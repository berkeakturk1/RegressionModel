import pandas as pd
import os

# Original dataset path
original_path = "/Users/berkeakturk/.cache/kagglehub/datasets/faruky/turkish-super-league-matches-19592020/versions/14/tsl_dataset.csv"

# Fixed dataset name
fixed_filename = "tsl_dataset_fixed.csv"

# Save to current working directory
fixed_path = os.path.join(os.getcwd(), fixed_filename)

# Mapping from original CSV to backend expected columns
column_rename_map = {
    'home': 'Home',
    'visitor': 'Away',
    'hgoal': 'HomeGoals',
    'vgoal': 'AwayGoals',
    'home_red_card': 'HomeRedCards',
    'visitor_red_card': 'AwayRedCards'
}

def fix_csv_structure():
    try:
        df = pd.read_csv(original_path)
        
        # Rename columns
        df = df.rename(columns=column_rename_map)
        
        # Ensure required columns exist now
        required_columns = ['Home', 'Away', 'HomeGoals', 'AwayGoals']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column missing after rename: {col}")

        # Fill optional columns if missing
        if 'HomeRedCards' not in df.columns:
            df['HomeRedCards'] = 0
        if 'AwayRedCards' not in df.columns:
            df['AwayRedCards'] = 0

        # Save the fixed dataset in current directory
        df.to_csv(fixed_path, index=False)
        print(f"✅ CSV structure fixed and saved to: {fixed_path}")

    except Exception as e:
        print(f"❌ Error fixing CSV: {str(e)}")

if __name__ == "__main__":
    fix_csv_structure()