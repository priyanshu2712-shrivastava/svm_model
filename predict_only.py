"""
predict_soil.py

Predict Soil Condition using pre-trained model

Usage:
python predict_soil.py --model soil_model.pkl --input new_data.csv --output predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import sys

def prepare_data(df):
    """Clean the dataframe by removing empty columns and index columns"""
    df = df.dropna(axis=1, how='all')
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c or c.lower() in ['index']]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df

def main():
    parser = argparse.ArgumentParser(description='Predict soil condition using trained model')
    parser.add_argument('--model', required=True, help='Path to trained model .pkl file')
    parser.add_argument('--input', required=True, help='Path to input CSV with features')
    parser.add_argument('--output', default='predictions.csv', help='Path for output CSV with predictions')
    args = parser.parse_args()
    
    try:
        # Load the trained model
        print(f"Loading model from {args.model}...")
        artifacts = joblib.load(args.model)
        
        # Extract components (matching the training script structure)
        model = artifacts['model']
        numeric_cols = artifacts['numeric_cols']
        cat_cols = artifacts['cat_cols']
        onehot_columns = artifacts['onehot_columns']
        imputer = artifacts['imputer']
        poly_transformer = artifacts.get('poly_transformer', None)
        
        print(f"✓ Model loaded successfully")
        print(f"  Numeric features: {len(numeric_cols)}")
        print(f"  Categorical features: {len(cat_cols)}")
        if poly_transformer:
            print(f"  Polynomial features: enabled")
        
        # Load input data
        print(f"\nLoading input data from {args.input}...")
        new_df = pd.read_csv(args.input, quotechar='"', skipinitialspace=True)
        
        # Remove trailing commas if present
        for col in new_df.columns:
            if new_df[col].dtype == 'object':
                new_df[col] = new_df[col].str.rstrip(',')
        
        original_df = new_df.copy()
        new_df = prepare_data(new_df)
        
        print(f"✓ Data loaded: {len(new_df)} rows, {len(new_df.columns)} columns")
        print(f"  Columns: {list(new_df.columns)}")
        
        # Process numeric features
        X_new_num = new_df.reindex(columns=numeric_cols).copy()
        missing_cols = [col for col in numeric_cols if col not in new_df.columns]
        if missing_cols:
            print(f"  Warning: Missing columns will be imputed: {missing_cols}")
        
        X_new_num = pd.DataFrame(
            imputer.transform(X_new_num),
            columns=numeric_cols,
            index=X_new_num.index
        )
        
        # Apply polynomial features if used during training
        if poly_transformer is not None:
            print(f"  Applying polynomial feature transformation...")
            X_new_num = pd.DataFrame(
                poly_transformer.transform(X_new_num),
                columns=poly_transformer.get_feature_names_out(numeric_cols),
                index=X_new_num.index
            )
        
        # Process categorical features
        if cat_cols:
            X_new_cat = new_df.reindex(columns=cat_cols).fillna('missing').astype(str)
            X_new_cat = pd.get_dummies(X_new_cat, drop_first=True)
            X_new_cat = X_new_cat.reindex(columns=onehot_columns, fill_value=0)
        else:
            X_new_cat = pd.DataFrame(index=new_df.index)
        
        # Combine features
        X_new_processed = pd.concat([X_new_num, X_new_cat], axis=1)
        
        print(f"  Final feature shape: {X_new_processed.shape}")
        
        # Make predictions
        print(f"\nMaking predictions...")
        
        predictions_soil = model.predict(X_new_processed)
        original_df['Predicted_Soil_Condition'] = predictions_soil
        print(f"  ✓ Soil Condition predicted")
        print(f"    Classes: {sorted(set(predictions_soil))}")
        
        # Save results
        original_df.to_csv(args.output, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS!")
        print(f"  Predictions saved to: {args.output}")
        print(f"  Total predictions: {len(original_df)}")
        print(f"{'='*60}")
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: File not found - {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"\n✗ ERROR: Missing key in model file - {e}")
        print("  This might be an old model file. Please retrain with the new script.")
        print(f"  Available keys in model: {list(artifacts.keys())}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()