"""
train_soil_condition.py

Enhanced soil condition prediction model

Key Features:
1. Feature engineering (polynomial features, interactions)
2. Feature selection to reduce noise
3. Multiple model types (RandomForest, XGBoost, SVM ensemble)
4. Better handling of imbalanced classes
5. Cross-validation with stratification
6. Optional SMOTE for severe imbalance
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")


def detect_soil_column(df):
    """Detect Soil Condition column"""
    soil_candidates = ['Soil Condition', 'Soil_Condition', 'soil_condition', 'Soil_Quality', 
                      'soil_quality', 'Soil Quality', 'quality', 'Quality', 'condition', 'Condition']
    
    soil_col = None
    
    for c in soil_candidates:
        if c in df.columns:
            soil_col = c
            break
    
    if not soil_col:
        for col in df.columns:
            if 'soil' in col.lower() and ('quality' in col.lower() or 'condition' in col.lower()):
                soil_col = col
                break
    
    return soil_col


def prepare_data(df):
    """Clean the dataframe"""
    df = df.dropna(axis=1, how='all')
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c or c.lower() in ['index']]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def analyze_target_distribution(y, target_name):
    """Analyze class distribution"""
    print(f"\n{'='*60}")
    print(f"{target_name} Distribution:")
    print(f"{'='*60}")
    value_counts = pd.Series(y).value_counts()
    print(value_counts)
    print(f"\nTotal classes: {len(value_counts)}")
    print(f"Min samples per class: {value_counts.min()}")
    print(f"Max samples per class: {value_counts.max()}")
    print(f"Imbalance ratio: {value_counts.max() / value_counts.min():.2f}:1")
    return value_counts


def create_feature_interactions(X_num):
    """Create interaction features for numeric columns"""
    if X_num.shape[1] < 2 or X_num.shape[1] > 10:
        return X_num, None
    
    print(f"Creating polynomial features (degree=2)...")
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_num)
    
    feature_names = poly.get_feature_names_out(X_num.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X_num.index)
    
    print(f"  Features increased: {X_num.shape[1]} → {X_poly_df.shape[1]}")
    return X_poly_df, poly


def build_ensemble_model(n_classes, use_smote=False, model_type='ensemble'):
    """Build ensemble model with multiple classifiers"""
    
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    
    elif model_type == 'ensemble':
        # Ensemble of multiple models
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        svm = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        estimators = [('rf', rf), ('svm', svm)]
        
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            estimators.append(('xgb', xgb))
        
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)


def train_soil_model(X_train, X_test, y_train, y_test, use_smote=False, 
                     feature_selection='mutual_info', model_type='ensemble'):
    """Train soil condition model"""
    
    print(f"\n{'='*60}")
    print(f"Training Soil Condition Model")
    print(f"{'='*60}")
    
    # Analyze distribution
    value_counts = analyze_target_distribution(y_train, "Soil Condition")
    
    # Determine number of features to select
    n_features = X_train.shape[1]
    k_features = min(int(n_features * 0.7), 100)  # Select 70% of features, max 100
    
    # Build pipeline components
    components = [('scaler', StandardScaler())]
    
    # Feature selection
    if feature_selection == 'mutual_info':
        components.append(('selector', SelectKBest(mutual_info_classif, k=k_features)))
    elif feature_selection == 'f_classif':
        components.append(('selector', SelectKBest(f_classif, k=k_features)))
    
    # SMOTE for imbalanced data
    if use_smote and SMOTE_AVAILABLE and value_counts.min() >= 6:
        # SMOTE requires at least 6 samples per class
        print(f"Applying SMOTE to balance classes...")
        components.append(('smote', SMOTE(random_state=42, k_neighbors=min(5, value_counts.min()-1))))
    
    # Add classifier
    classifier = build_ensemble_model(len(value_counts), use_smote, model_type)
    components.append(('classifier', classifier))
    
    # Build pipeline
    if use_smote and SMOTE_AVAILABLE:
        pipeline = ImbPipeline(components)
    else:
        pipeline = Pipeline(components)
    
    # Train
    print(f"Training with {model_type} model...")
    pipeline.fit(X_train, y_train)
    
    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Test evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nSoil Condition Test Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance if available
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].feature_importances_
        print(f"\nTop 10 Most Important Features:")
        # Get feature names after selection
        if 'selector' in pipeline.named_steps:
            mask = pipeline.named_steps['selector'].get_support()
            selected_features = X_train.columns[mask]
            feature_imp = pd.Series(importances, index=selected_features).sort_values(ascending=False)
        else:
            feature_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        print(feature_imp.head(10))
    
    return pipeline, acc, f1


def main(args):
    print("="*60)
    print("SOIL CONDITION PREDICTION MODEL")
    print("="*60)
    
    # Load data
    df = pd.read_csv(args.train_csv)
    df = prepare_data(df)
    print(f"\nLoaded data shape: {df.shape}")
    
    # Detect soil condition column
    soil_col = detect_soil_column(df)
    print(f"\nDetected Soil Condition column: '{soil_col}'")
    
    if not soil_col:
        raise ValueError("Could not detect soil condition column.")
    
    # Prepare features
    X = df.drop(columns=[soil_col])
    y_soil = df[soil_col]
    
    # Handle categorical features
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nFeature summary:")
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    
    # Process numeric features with imputation
    X_num = X[numeric_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_num = pd.DataFrame(imputer.fit_transform(X_num), 
                        columns=numeric_cols, index=X.index)
    
    # Create polynomial features if enabled
    poly_transformer = None
    if args.poly_features and len(numeric_cols) > 1:
        X_num, poly_transformer = create_feature_interactions(X_num)
    
    # Process categorical features
    if cat_cols:
        X_cat = X[cat_cols].fillna('missing').astype(str)
        X_cat = pd.get_dummies(X_cat, drop_first=True)
        print(f"  One-hot encoded features: {X_cat.shape[1]}")
    else:
        X_cat = pd.DataFrame(index=X.index)
    
    X_processed = pd.concat([X_num, X_cat], axis=1)
    print(f"\nFinal feature shape: {X_processed.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_soil, test_size=0.2, random_state=42,
        stratify=y_soil if len(np.unique(y_soil)) > 1 else None
    )
    
    # Train model
    model, acc, f1 = train_soil_model(
        X_train, X_test, y_train, y_test,
        use_smote=args.use_smote,
        feature_selection=args.feature_selection,
        model_type=args.model_type
    )
    
    # Save artifacts
    artifacts = {
        'model': model,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols,
        'onehot_columns': X_cat.columns.tolist(),
        'imputer': imputer,
        'poly_transformer': poly_transformer,
        'soil_column': soil_col,
        'metrics': {'accuracy': acc, 'f1_score': f1}
    }
    
    joblib.dump(artifacts, args.model_out)
    print(f"\n{'='*60}")
    print(f"✓ Model and artifacts saved to {args.model_out}")
    print(f"\nFinal Performance Summary:")
    print(f"  Soil Condition: Accuracy={acc:.4f}, F1={f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Soil condition prediction model')
    parser.add_argument('--train_csv', required=True, help='Training CSV path')
    parser.add_argument('--model_out', default='soil_model.pkl', help='Model output path')
    parser.add_argument('--use_smote', action='store_true', help='Use SMOTE for imbalanced classes')
    parser.add_argument('--poly_features', action='store_true', help='Create polynomial features')
    parser.add_argument('--feature_selection', default='mutual_info', 
                       choices=['mutual_info', 'f_classif', 'none'], 
                       help='Feature selection method')
    parser.add_argument('--model_type', default='ensemble',
                       choices=['ensemble', 'random_forest', 'xgboost'],
                       help='Model type to use')
    
    args = parser.parse_args()
    main(args)