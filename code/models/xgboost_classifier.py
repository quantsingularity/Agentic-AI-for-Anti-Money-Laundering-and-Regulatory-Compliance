"""
XGBoost Crime Typology Classifier
Supervised classification of suspicious transactions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import pickle


class XGBoostClassifier:
    """
    XGBoost-based classifier for fraud detection and typology classification.
    
    Features:
    - Engineered features from transaction data
    - Multi-class and binary classification
    - Probability calibration
    - Feature importance analysis
    """
    
    def __init__(self, task='binary', seed=42):
        """
        Initialize classifier.
        
        Args:
            task: 'binary' for fraud/benign, 'multiclass' for typology
            seed: Random seed
        """
        self.task = task
        self.seed = seed
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task == 'multiclass' else None
        self.model = None
        self.feature_names = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from transaction data.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        features = df.copy()
        
        # Basic features
        features['amount_log'] = np.log1p(features['amount'])
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        
        # Encode categorical
        features['sender_country_enc'] = features['sender_country'].astype('category').cat.codes
        features['receiver_country_enc'] = features['receiver_country'].astype('category').cat.codes
        features['transaction_type_enc'] = features['transaction_type'].astype('category').cat.codes
        
        # Cross-border indicator
        features['is_cross_border'] = (features['sender_country'] != features['receiver_country']).astype(int)
        
        # Aggregate features per sender (requires groupby)
        sender_stats = df.groupby('sender_id').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'transaction_id': 'count'
        }).reset_index()
        sender_stats.columns = ['sender_id', 'sender_txn_count', 'sender_total_amount', 
                               'sender_mean_amount', 'sender_std_amount', 'sender_txn_count2']
        sender_stats['sender_std_amount'] = sender_stats['sender_std_amount'].fillna(0)
        
        features = features.merge(sender_stats[['sender_id', 'sender_txn_count', 'sender_total_amount',
                                               'sender_mean_amount', 'sender_std_amount']], 
                                 on='sender_id', how='left')
        
        # Receiver stats
        receiver_stats = df.groupby('receiver_id').agg({
            'amount': ['count', 'sum', 'mean'],
        }).reset_index()
        receiver_stats.columns = ['receiver_id', 'receiver_txn_count', 'receiver_total_amount', 
                                 'receiver_mean_amount']
        
        features = features.merge(receiver_stats, on='receiver_id', how='left')
        
        # Fill NaN for single transactions
        for col in ['sender_txn_count', 'sender_total_amount', 'sender_mean_amount', 'sender_std_amount',
                   'receiver_txn_count', 'receiver_total_amount', 'receiver_mean_amount']:
            if col in features.columns:
                features[col] = features[col].fillna(0)
        
        # Velocity features: amount deviation from sender average
        features['amount_vs_sender_avg'] = features['amount'] / (features['sender_mean_amount'] + 1)
        
        # High-risk country indicators
        high_risk = ['SY', 'IR', 'KP', 'VE', 'MM', 'AF', 'IQ', 'PA', 'KY', 'BZ', 'VG']
        features['sender_high_risk'] = features['sender_country'].isin(high_risk).astype(int)
        features['receiver_high_risk'] = features['receiver_country'].isin(high_risk).astype(int)
        
        return features
    
    def prepare_data(self, features: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target.
        
        Args:
            features: Feature DataFrame
            target_col: Target column name
            
        Returns:
            X, y arrays
        """
        # Select numeric features
        feature_cols = [
            'amount', 'amount_log', 'hour', 'day_of_week',
            'sender_country_enc', 'receiver_country_enc', 'transaction_type_enc',
            'is_cross_border', 'sender_txn_count', 'sender_total_amount',
            'sender_mean_amount', 'sender_std_amount', 'receiver_txn_count',
            'receiver_total_amount', 'receiver_mean_amount', 'amount_vs_sender_avg',
            'sender_high_risk', 'receiver_high_risk'
        ]
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in features.columns]
        self.feature_names = feature_cols
        
        X = features[feature_cols].values
        y = features[target_col].values
        
        # Encode labels for multiclass
        if self.task == 'multiclass':
            y = self.label_encoder.fit_transform(y)
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # XGBoost parameters
        if self.task == 'binary':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': self.seed,
                'tree_method': 'hist'
            }
        else:  # multiclass
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y_train)),
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': self.seed,
                'tree_method': 'hist'
            }
        
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train, verbose=False)
        
        return {'status': 'trained'}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dict with metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        results = {
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        if self.task == 'binary':
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            results.update({
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'roc_auc': roc_auc_score(y, y_proba[:, 1]) if y_proba.shape[1] > 1 else 0.5
            })
            
            # PR curve
            precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba[:, 1])
            results['pr_auc'] = auc(recall_curve, precision_curve)
            
        else:  # multiclass
            from sklearn.metrics import accuracy_score
            results['accuracy'] = accuracy_score(y, y_pred)
            results['classification_report'] = classification_report(y, y_pred, output_dict=True)
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """Save model to file."""
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'task': self.task
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.label_encoder = model_dict['label_encoder']
        self.feature_names = model_dict['feature_names']
        self.task = model_dict['task']
