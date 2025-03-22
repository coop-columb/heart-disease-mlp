"""
Data preprocessing module for the Heart Disease Prediction project.
"""

import logging
from typing import List, Tuple

# numpy used in development
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from specified path.

    Args:
        data_path: Path to the dataset

    Returns:
        DataFrame containing the dataset
    """
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def binarize_target(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """
    Convert target variable to binary (0/1).

    Args:
        df: DataFrame containing the dataset
        target_col: Name of the target column

    Returns:
        DataFrame with binarized target
    """
    logger.info(f"Binarizing target column: {target_col}")
    df = df.copy()

    # Convert to binary if not already
    if df[target_col].nunique() > 2:
        logger.info(f"Converting {target_col} from multiclass to binary")
        df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)

    logger.info(
        f"Target distribution after binarization: {df[target_col].value_counts().to_dict()}"
    )
    return df


# This function is intentionally complex due to the specific rules for each feature
# noqa: C901
def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: DataFrame containing the dataset
        numeric_strategy: Strategy for imputing numerical values
        categorical_strategy: Strategy for imputing categorical values

    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values")
    df = df.copy()

    # Get column types
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Log missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info("Missing values found:")
        for col in missing_values[missing_values > 0].index:
            logger.info(
                f"  {col}: {missing_values[col]} ({missing_values[col]/len(df)*100:.2f}%)"
            )

        # Impute missing values
        logger.info(f"Imputing numerical values with {numeric_strategy}")
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if numeric_strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_strategy == "knn":
                    # This is a placeholder, actual KNN imputation will be handled by the pipeline
                    logger.info(
                        "KNN imputation will be handled by the preprocessing pipeline"
                    )
                else:
                    raise ValueError(
                        f"Unknown numeric imputation strategy: {numeric_strategy}"
                    )

        logger.info(f"Imputing categorical values with {categorical_strategy}")
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if categorical_strategy == "most_frequent":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    raise ValueError(
                        f"Unknown categorical imputation strategy: {categorical_strategy}"
                    )
    else:
        logger.info("No missing values found in the dataset")

    return df


def create_preprocessing_pipeline(
    categorical_features: List[str],
    numerical_features: List[str],
    numeric_imputer: str = "median",
    use_robust_scaler: bool = True,
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline using scikit-learn.

    Args:
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        numeric_imputer: Strategy for imputing numerical values ('median', 'mean', 'knn')
        use_robust_scaler: Whether to use RobustScaler instead of StandardScaler

    Returns:
        ColumnTransformer preprocessing pipeline
    """
    logger.info("Creating preprocessing pipeline")

    # Configure numerical preprocessor
    if numeric_imputer == "knn":
        logger.info("Using KNN imputer for numerical features")
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        logger.info(f"Using {numeric_imputer} imputer for numerical features")
        num_imputer = SimpleImputer(strategy=numeric_imputer)

    # Select scaler
    if use_robust_scaler:
        logger.info("Using RobustScaler for numerical features")
        scaler = RobustScaler()
    else:
        logger.info("Using StandardScaler for numerical features")
        scaler = StandardScaler()

    # Create transformers
    numerical_transformer = Pipeline(
        steps=[("imputer", num_imputer), ("scaler", scaler)]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Preprocessing pipeline created successfully")
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature data
        y: Target data
        test_size: Proportion of data for the test set
        val_size: Proportion of training data for the validation set
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}")

    # First split: training+validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    logger.info(
        f"Data split complete. Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
