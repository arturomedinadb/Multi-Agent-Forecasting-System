"""
Training functions for demand forecasting models.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import catboost as cb

from agents import function_tool
from ..schemas.forecasting_models import ModelConfig, AllModelConfigs, ModelType, HyperparameterTuningParameters

# Supported models
SUPPORTED_MODELS = {
    "xgboost": xgb.XGBRegressor,
    "random_forest": RandomForestRegressor,
    "lightgbm": lgb.LGBMRegressor,
    "catboost": cb.CatBoostRegressor
}

@function_tool
async def create_model_configs(
    model_types: List[str],
    hyperparameters: Dict[str, Any] = None,
    iteration: int = 1
) -> AllModelConfigs:
    """Create model configurations for the specified model types."""
    try:
        configs = []
        session_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        for model_type_str in model_types:
            if model_type_str in SUPPORTED_MODELS:
                # Convert string to ModelType enum
                model_type_enum = ModelType[model_type_str.upper().replace("-", "_")]
                
                model_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                
                try:
                    hyperparameters = hyperparameters[model_type_str]
                except:
                    hyperparameters = get_default_hyperparameters(model_type_str)

                config = ModelConfig(
                    model_type=model_type_enum,
                    model_name=f"{model_type_str}_model_{model_timestamp}",
                    model_uuid=model_timestamp,
                    hyperparameters=hyperparameters
                )
                configs.append(config)
                print(f"Created model configuration: {config.model_name}")
        
        result = AllModelConfigs(
            configs=configs,
            total_configs=len(configs),
            iteration=iteration,
            session_uuid=session_timestamp
        )
        
        return result
        
    except Exception as e:
        # Return a minimal valid AllModelConfigs on error
        return AllModelConfigs(
            configs=[],
            total_configs=0,
            iteration=iteration,
            session_uuid=datetime.now().strftime("%Y%m%d%H%M%S")
        )

@function_tool
async def load_and_preprocess_data(
    input_file: str,
    output_dir: str,
    id_column: list[str],
    target_column: str = "units_sold"
) -> Dict[str, Any]:
    """Load and preprocess data, saving processed datasets."""
    try:
        # Load data
        df = pd.read_csv(input_file)
        print(f"\n{datetime.now()} - Loaded data shape: {df.shape}")

        df = df.sort_values(by=id_column, ascending=True, ignore_index=True)

        # Save original ID column values before encoding
        original_id_data = {}
        for col in id_column:
            if col in df.columns:
                original_id_data[col] = df[col].copy()
        
        # Handle id columns - encode them for model training
        for col in id_column:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # Handle missing values for numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != target_column and col not in id_column:
                df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != target_column:
                # Check if it's a date column
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='raise')
                        df = df.drop(col, axis=1)  # Drop original date column
                        
                    except:
                        # If not a date, treat as categorical
                        df[col] = pd.Categorical(df[col]).codes
        
        # Handle missing values for target column
        if target_column in df.columns:
            df[target_column] = df[target_column].fillna(df[target_column].median())
        
        # Ensure target column is numeric
        if target_column in df.columns:
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            df[target_column] = df[target_column].fillna(df[target_column].median())
        
        # Remove any remaining non-numeric columns except target
        for col in df.columns:
            if col != target_column and not pd.api.types.is_numeric_dtype(df[col]):
                df = df.drop(col, axis=1)
        
        # Split data
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Use TimeSeriesSplit for all sets (train, validation, test)
        print(f"Using TimeSeriesSplit for all data splits")
        total_size = len(X)
        
        # Use TimeSeriesSplit to get train and test (50-50 split)
        tscv = TimeSeriesSplit(n_splits=4)
        splits = list(tscv.split(X))
        
        # Use split 1: 80% train, 20% test
        train_val_idx, test_idx = splits[3]
        
        X_train_val = X.iloc[train_val_idx]
        y_train_val = y.iloc[train_val_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Now split the train portion (80%) into train (64%) and val (16%)
        tscv_sub = TimeSeriesSplit(n_splits=4)
        splits_sub = list(tscv_sub.split(X_train_val))
        train_idx, val_idx = splits_sub[3]

        X_val = X_train_val.iloc[val_idx]
        y_val = y_train_val.iloc[val_idx]
        X_train = X_train_val.iloc[train_idx]
        y_train = y_train_val.iloc[train_idx]
        
        # Extract original ID values for each split
        # train_idx and val_idx are relative to train_val_idx, so we need to map them back
        original_id_train = {}
        original_id_val = {}
        original_id_test = {}
        for col in id_column:
            if col in original_id_data:
                # Map indices: train_idx/val_idx are relative to train_val_idx
                train_absolute_idx = train_val_idx[train_idx]
                val_absolute_idx = train_val_idx[val_idx]
                original_id_train[col] = original_id_data[col].iloc[train_absolute_idx].values
                original_id_val[col] = original_id_data[col].iloc[val_absolute_idx].values
                original_id_test[col] = original_id_data[col].iloc[test_idx].values
        
        print(f"Split sizes - Train: {len(X_train)} ({len(X_train)/total_size*100:.1f}%), Val: {len(X_val)} ({len(X_val)/total_size*100:.1f}%), Test: {len(X_test)} ({len(X_test)/total_size*100:.1f}%)")
        print(f"TimeSeriesSplit - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
            
        # Create output directory
        data_dir = f"{output_dir}/data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save processed data
        pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"{data_dir}/X_train.csv", index=False)
        pd.DataFrame(X_val_scaled, columns=X.columns).to_csv(f"{data_dir}/X_val.csv", index=False)
        pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"{data_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{data_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{data_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{data_dir}/y_test.csv", index=False)
        
        # Save scaler and feature names
        joblib.dump(scaler, f"{data_dir}/scaler.pkl")
        with open(f"{data_dir}/feature_names.json", "w") as f:
            json.dump(list(X.columns), f)
        
        # Save ID column names for later use in predictions
        with open(f"{data_dir}/id_columns.json", "w") as f:
            json.dump(id_column, f)
        
        # Save original ID values for each split
        for col in id_column:
            if col in original_id_test:
                pd.DataFrame({col: original_id_test[col]}).to_csv(f"{data_dir}/id_test_{col}.csv", index=False)
                pd.DataFrame({col: original_id_train[col]}).to_csv(f"{data_dir}/id_train_{col}.csv", index=False)
                pd.DataFrame({col: original_id_val[col]}).to_csv(f"{data_dir}/id_val_{col}.csv", index=False)
        
        print(f"\n{datetime.now()} - Data preprocessing completed successfully")
        
        return {
            "success": True,
            "data_dir": data_dir,
            "shape": df.shape,
            "features": len(X.columns),
            "feature_names": list(X.columns),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "split_method": "time_series",
            "message": "Data preprocessing completed successfully using time series split"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error in data preprocessing: {str(e)}"
        }

@function_tool
async def train_models(
    all_model_configs: AllModelConfigs,
    data_dir: str,
    output_dir: str
) -> Dict[str, Any]:
    """Train all models with the given configurations."""
    print(f"\n{datetime.now()} - Training {all_model_configs.total_configs} models at iteration {all_model_configs.iteration}")
    try:
        # Load data
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_val = pd.read_csv(f"{data_dir}/X_val.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
        y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.ravel()
        
        session_uuid = all_model_configs.session_uuid
        model_configs = all_model_configs.configs
        
        results = {
            "success": True,
            "models": [],
            "best_model": None,
            "best_score": float('inf'),
            "training_time": 0,
            "models_directory": f"{output_dir}/models/{session_uuid}",
            "data_dir": data_dir,  # Add data_dir for evaluation agent
            "session_uuid": session_uuid,
            "training_timestamp": datetime.now().isoformat(),
            "total_models": len(model_configs),  # Add total_models count
            "message": f"Trained {len(model_configs)} models successfully"
        }
        
        start_time = datetime.now()
        
        for config in model_configs:
            try:
                model_result = await train_single_model(config, X_train, X_val, y_train, y_val, output_dir, session_uuid)
                results["models"].append(model_result)
                
                # Track best model
                if model_result["success"] and model_result["rmse"] < results["best_score"]:
                    results["best_score"] = model_result["rmse"]
                    results["best_model"] = model_result["model_name"]
        
            except Exception as e:
                print(f"Error training {config.model_name}: {str(e)}")
                results["models"].append({
                    "success": False,
                    "model_name": config.model_name,
                    "error": str(e)
                })
        
        results["training_time"] = (datetime.now() - start_time).total_seconds()
        
        return results
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
    return {
            "success": False,
            "error": str(e),
            "message": f"Error in model training: {str(e)}"
        }

@function_tool
async def apply_feature_engineering(
    original_file: str,
    output_dir: str,
    suggestions: List[str],
    iteration: int,
    target_column: str = "units_sold"
) -> Dict[str, Any]:
    """Apply feature engineering based on suggestions."""
    try:
        print(f"\n{datetime.now()} - Applying feature engineering based on suggestions: {suggestions} at iteration {iteration}")
        # Load original data
        df = pd.read_csv(original_file)
        
        # Apply feature engineering suggestions
        for suggestion in suggestions:
            if suggestion == "polynomial_features":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != target_column:
                        df[f"{col}_squared"] = df[col] ** 2
            elif suggestion == "log_transformation":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != target_column and df[col].min() > 0:
                        df[f"{col}_log"] = np.log1p(df[col])
            elif suggestion == "interaction_features":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != target_column]
                if len(numeric_cols) >= 2:
                    df[f"{numeric_cols[0]}_x_{numeric_cols[1]}"] = df[numeric_cols[0]] * df[numeric_cols[1]]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
        # Save new dataset with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_file = f"{output_dir}/feature_engineered_{timestamp}.csv"
        df.to_csv(new_file, index=False)
        
        print(f"\n{datetime.now()} - Saved feature engineered dataset to: {new_file}")
        
        return {
            "success": True,
            "new_input_file": new_file,
            "output_dir": output_dir,
            "original_shape": df.shape,
            "new_features": len(df.columns),
            "message": f"Feature engineering applied successfully. New file: {new_file}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error in feature engineering: {str(e)}"
        }

@function_tool
async def apply_hyperparameter_tuning(
    model_type: str,
    target_model: str,
    parameters: HyperparameterTuningParameters,
    data_dir: str,
    output_dir: str,
    iteration: int
) -> Dict[str, Any]:
    """
    Apply hyperparameter tuning to a specific model using validation set.
    
    Args:
        model_type: Type of model (xgboost, random_forest, lightgbm, catboost)
        target_model: Name of the target model to tune
        parameters: HyperparameterTuningParameters schema containing:
            - hyperparameters: Dict[str, List[Union[int, float, str]]] - hyperparameter grid to search
            - optimization_method: str - method to use (default: grid_search)
            - n_trials: Optional[int] - number of trials (for random search)
        data_dir: Directory containing train/val/test splits
        output_dir: Output directory for saving tuned model
        iteration: Current iteration number
        
    Returns:
        Dict with tuned model information including model_path
    """
    try:
        print(f"\n{datetime.now()} - Applying hyperparameter tuning at iteration {iteration} to {target_model}")
        print(f"Parameters received: {parameters}")
        
        # Load data - use validation set for hyperparameter tuning evaluation
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_val = pd.read_csv(f"{data_dir}/X_val.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
        y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.ravel()
        
        # Extract validated hyperparameters from schema
        hyperparameter_grid = parameters.hyperparameters
        optimization_method = parameters.optimization_method
        n_trials = parameters.n_trials
        
        print(f"Hyperparameter grid: {hyperparameter_grid}")
        print(f"Optimization method: {optimization_method}")
        
        # Get base model class
        model_class = SUPPORTED_MODELS.get(model_type)
        if not model_class:
            return {
                "success": False,
                "error": f"Unsupported model type: {model_type}",
                "message": f"Model type {model_type} is not supported"
            }
        
        # Get default hyperparameters and merge with grid
        default_params = get_default_hyperparameters(model_type)
        param_grid = {}
        
        # Build parameter grid from feedback
        for param_name, param_values in hyperparameter_grid.items():
            if isinstance(param_values, list):
                param_grid[param_name] = param_values
            else:
                # If single value, create a list
                param_grid[param_name] = [param_values]
        
        # Add default parameters that aren't in the grid
        for param_name, param_value in default_params.items():
            if param_name not in param_grid:
                param_grid[param_name] = [param_value]
        
        print(f"Final parameter grid: {param_grid}")
        
        # Perform hyperparameter tuning using TimeSeriesSplit
        start_time = datetime.now()
        
        if optimization_method == "grid_search" and param_grid:
            # Use GridSearchCV with TimeSeriesSplit for time series cross-validation
            # Combine train and val for TimeSeriesSplit
            X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
            y_train_val = np.concatenate([y_train, y_val])
            
            # Create TimeSeriesSplit for time series cross-validation
            # Use 3 splits to ensure we have enough data for training and validation
            n_splits = 3
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Create base model with fixed parameters (not in grid)
            base_params = {k: (v[0] if isinstance(v, list) else v) for k, v in default_params.items() if k not in param_grid}
            
            # Create base model class with fixed parameters
            base_model = model_class(**base_params)
            
            # Use RMSE as scoring metric (lower is better)
            rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
            
            # Use GridSearchCV with TimeSeriesSplit
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=rmse_scorer,
                cv=tscv,
                n_jobs=-1,  # Use all available cores
                verbose=1,
                refit=False  # Don't refit - we'll train on training set only after finding best params
            )
            
            print(f"\n{datetime.now()} - Starting GridSearchCV with TimeSeriesSplit ({n_splits} splits)")
            print(f"Parameter grid: {param_grid}")
            print(f"Total data points for cross-validation: {len(X_train_val)}")
            
            # Fit GridSearchCV (this will use TimeSeriesSplit for cross-validation)
            grid_search.fit(X_train_val, y_train_val)
            
            # Get best parameters
            best_params_grid = grid_search.best_params_
            
            # Merge best_params_grid with base_params
            best_params = {**base_params, **best_params_grid}
            
            # With greater_is_better=False, best_score_ is the actual metric value (RMSE)
            # GridSearchCV optimizes the negated version internally but returns the actual value
            best_score = grid_search.best_score_
            
            # Train final model on training set only using best parameters
            print(f"\n{datetime.now()} - Training final model on training set with best parameters")
            best_model = model_class(**best_params)
            best_model.fit(X_train, y_train)
            
            # Evaluate on validation set to get final validation RMSE
            y_pred_val = best_model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            
            print(f"\n{datetime.now()} - GridSearchCV completed")
            print(f"Best parameters: {best_params}")
            print(f"Best CV RMSE: {best_score:.4f}")
            print(f"Final validation RMSE (on training set only): {val_rmse:.4f}")
                    
        else:
            # If no grid provided or method is not grid_search, use default with single trial
            print("No hyperparameter grid provided or method not grid_search, using default parameters")
            best_params = default_params
            best_model = model_class(**default_params)
            best_model.fit(X_train, y_train)
            y_pred_val = best_model.predict(X_val)
            best_score = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions on both sets with best model
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        print(f"\n{datetime.now()} - Best hyperparameters: {best_params}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        
        # Generate timestamp for tuned model
        tuned_model_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        tuned_model_name = f"{target_model}_tuned_{tuned_model_timestamp}"
        
        # Extract session UUID from target_model or use current timestamp
        # target_model format: "xgboost_model_20241215143022"
        model_timestamp = target_model.split("_")[-1] if "_" in target_model else datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Get session UUID from output_dir structure or use model_timestamp
        # Try to find existing session directory
        models_base_dir = f"{output_dir}/models"
        session_uuid = model_timestamp
        
        # Save model with timestamp in filename
        model_dir = f"{output_dir}/models/{session_uuid}"
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"{tuned_model_name}.pkl"
        model_path = f"{model_dir}/{model_filename}"
        joblib.dump(best_model, model_path)
        
        print(f"\n{datetime.now()} - Tuned model saved to: {model_path}")
        
        # Save model config as JSON
        metrics = {
            "rmse": val_rmse,
            "r2": val_r2,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "training_time": training_time
        }
        additional_metadata = {
            "original_model": target_model,
            "optimization_method": optimization_method,
            "best_cv_score": best_score if optimization_method == "grid_search" else None
        }
        save_model_config(
            model_dir=model_dir,
            model_name=tuned_model_name,
            model_type=model_type,
            model_uuid=tuned_model_timestamp,
            hyperparameters=best_params,
            model_path=model_path,
            metrics=metrics,
            additional_metadata=additional_metadata
        )
        
        return {
            "success": True,
            "model_name": tuned_model_name,
            "model_type": model_type,
            "model_uuid": tuned_model_timestamp,
            "model_filename": model_filename,
            "original_model": target_model,
            "hyperparameters": best_params,
            "rmse": val_rmse,
            "r2": val_r2,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "training_time": training_time,
            "model_path": model_path,
            "created_at": datetime.now().isoformat(),
            "message": f"Hyperparameter tuning completed for {target_model} using validation set"
        }
        
    except Exception as e:
        print(f"\n{datetime.now()} - Error in hyperparameter tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"Error in hyperparameter tuning: {str(e)}"
        }

# Helper functions
def save_model_config(
    model_dir: str,
    model_name: str,
    model_type: str,
    model_uuid: str,
    hyperparameters: Dict[str, Any],
    model_path: str,
    metrics: Dict[str, Any] = None,
    additional_metadata: Dict[str, Any] = None
) -> str:
    """
    Save model configuration as JSON file.
    
    Args:
        model_dir: Directory where model is saved
        model_name: Name of the model
        model_type: Type of model (xgboost, lightgbm, etc.)
        model_uuid: Unique identifier for the model
        hyperparameters: Model hyperparameters
        model_path: Path to the saved model pkl file
        metrics: Optional dictionary of training metrics
        additional_metadata: Optional dictionary of additional metadata
        
    Returns:
        Path to the saved JSON config file
    """
    config_data = {
        "model_name": model_name,
        "model_type": model_type,
        "model_uuid": model_uuid,
        "hyperparameters": hyperparameters,
        "model_path": model_path,
        "created_at": datetime.now().isoformat(),
    }
    
    # Add metrics if provided
    if metrics:
        config_data["metrics"] = metrics
    
    # Add additional metadata if provided
    if additional_metadata:
        config_data.update(additional_metadata)
    
    # Save to JSON file
    config_filename = f"{model_name}_metadata.json"
    config_path = f"{model_dir}/{config_filename}"
    
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2, default=str)
    
    print(f"{datetime.now()} - Model config saved to: {config_path}")
    return config_path

def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
    """Get default hyperparameters for a model type."""
    defaults = {
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "catboost": {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3,
            "random_state": 42,
            "verbose": False
        }
    }
    return defaults.get(model_type, {})

async def train_single_model(
    config: ModelConfig,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    output_dir: str,
    session_uuid: str
) -> Dict[str, Any]:
    """Train a single model."""
    
    model_type = config.model_type.value if hasattr(config.model_type, 'value') else config.model_type
    model_name = config.model_name
    hyperparams = config.hyperparameters
    
    # Create model
    model_class = SUPPORTED_MODELS[model_type]
    model = model_class(**hyperparams)
    
    # Train model
    start_time = datetime.now()
    print(f"\n{start_time} -  Training model {model_name} with hyperparameters: {hyperparams}")
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Model {model_name} completed in {training_time} seconds")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    
    # Generate model timestamp for this specific model
    model_timestamp = config.model_uuid
    
    # Save model with timestamp in filename
    model_dir = f"{output_dir}/models/{session_uuid}"
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"{model_name}.pkl"
    model_path = f"{model_dir}/{model_filename}"
    joblib.dump(model, model_path)
    
    # Save model config as JSON
    metrics = {
        "rmse": val_rmse,
        "r2": val_r2,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "training_time": training_time
    }
    save_model_config(
        model_dir=model_dir,
        model_name=model_name,
        model_type=model_type,
        model_uuid=model_timestamp,
        hyperparameters=hyperparams,
        model_path=model_path,
        metrics=metrics
    )
    
    return {
        "success": True,
        "model_name": model_name,
        "model_type": model_type,
        "model_uuid": model_timestamp,
        "model_filename": model_filename,
        "hyperparameters": hyperparams,
        "rmse": val_rmse,
        "r2": val_r2,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "training_time": training_time,
        "model_path": model_path,
        "created_at": datetime.now().isoformat()
    }

@function_tool
async def train_ensemble_models(
    base_models: List[Dict[str, Any]],
    data_dir: str,
    output_dir: str,
    iteration: int,
    ensemble_method: str = "voting"
) -> Dict[str, Any]:
    """Train ensemble models using the specified base models."""
    print(f"\n{datetime.now()} - Training ensemble models at iteration {iteration} using the specified base models: {base_models}")
    try:
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import LinearRegression
        
        # Load data
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_val = pd.read_csv(f"{data_dir}/X_val.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
        y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.ravel()
        
        # Load base models
        models_directory = f"{output_dir}/models"
        base_estimators = []
        
        for model_info in base_models:
            model_name = model_info["model_name"]
            model_path = f"{models_directory}/{model_name}.pkl"
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                base_estimators.append((model_name, model))
            else:
                print(f"Warning: Model {model_name} not found at {model_path}")
        
        if not base_estimators:
            return {
                "success": False,
                "error": "No base models found",
                "message": "Cannot create ensemble - no base models available"
            }
        
        # Create ensemble model
        if ensemble_method == "voting":
            ensemble_model = VotingRegressor(base_estimators)
        else:
            # Default to voting if method not recognized
            ensemble_model = VotingRegressor(base_estimators)
        
        # Train ensemble
        start_time = datetime.now()
        ensemble_model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate ensemble
        y_pred_train = ensemble_model.predict(X_train)
        y_pred_val = ensemble_model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        # Generate timestamp for ensemble model
        ensemble_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ensemble_name = f"ensemble_{ensemble_method}_{ensemble_timestamp}"
        
        # Save ensemble model
        model_dir = f"{output_dir}/models"
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"{ensemble_name}.pkl"
        model_path = f"{model_dir}/{model_filename}"
        joblib.dump(ensemble_model, model_path)
        
        # Save model config as JSON
        metrics = {
            "rmse": val_rmse,
            "r2": val_r2,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "training_time": training_time
        }
        additional_metadata = {
            "ensemble_method": ensemble_method,
            "base_models": [model[0] for model in base_estimators],
            "num_base_models": len(base_estimators)
        }
        # Ensemble models don't have hyperparameters in the traditional sense
        save_model_config(
            model_dir=model_dir,
            model_name=ensemble_name,
            model_type="ensemble",
            model_uuid=ensemble_timestamp,
            hyperparameters={},  # Ensemble models don't have hyperparameters
            model_path=model_path,
            metrics=metrics,
            additional_metadata=additional_metadata
        )
    
        return {
            "success": True,
            "model_name": ensemble_name,
            "model_type": "ensemble",
            "model_uuid": ensemble_timestamp,
            "model_filename": model_filename,
            "ensemble_method": ensemble_method,
            "base_models": [model[0] for model in base_estimators],
            "rmse": val_rmse,
            "r2": val_r2,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "training_time": training_time,
            "model_path": model_path,
            "created_at": datetime.now().isoformat(),
            "message": f"Ensemble model trained with {len(base_estimators)} base models"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error training ensemble model: {str(e)}"
        }