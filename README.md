# Iterative Feature Selector (ITFS)

Iterative Feature Selector (IFS) is a Python package designed to help with feature selection in machine learning projects. It provides tools to iteratively select and drop features based on various criteria such as fixed value columns, high correlation, null importance, and feature importance. Deciding on thresholds for these criteria can be challenging, so this package incorporates statistical tests to aid in the selection process. Additionally, it allows for iterative testing of various thresholds to optimize feature selection. ITFS stands for Iterative Feature Selection.

## Installation

To install the package, use the following command:

```sh
pip install itfs
```

## Usage

### Example

Here is an example of how to use the Iterative Feature Selector:

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from itfs import IterativeFeatureSelector

# Create a sample dataset
def create_sample_data():
    from sklearn.datasets import fetch_california_housing
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name='target')
    df = pd.concat([y, X], axis=1)
    return df

# Define error and importance calculation function
def calc_error_importance(xdf: pd.DataFrame, ys: np.ndarray) -> tuple[list[float], pd.DataFrame]:
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    errors = []
    importance_df = []
    for train_idx, test_idx in kf.split(xdf):
        train_x, test_x = xdf.iloc[train_idx], xdf.iloc[test_idx]
        train_y, test_y = ys[train_idx], ys[test_idx]
        model = lgb.LGBMRegressor()
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        errors.append(mean_squared_error(test_y, y_pred, squared=False))
        importance_df.append(pd.DataFrame(model.feature_importances_, index=xdf.columns, columns=['importance']))
    importance_df = pd.concat(importance_df, axis=1)
    return errors, importance_df

# Load data
df = create_sample_data()

# Initialize IterativeFeatureSelector
ifs = IterativeFeatureSelector(
    force_drop_cols=["drop1"],
    ignore_cols=["ignore1"],
    ycol="target",
    min_col_size=2,
)

# Fit and select features
ifs.fit_const_cols(df)
ifs.fit_corr_cols(df, cor_th=0.9, error_func=calc_error_importance)
ifs.fit_null_cols(df, null_p_th=0.05, error_importance_func=calc_error_importance)
ifs.fit_feature_importance_cols(df, importance_p_th=0.01, error_importance_func=calc_error_importance)

# Get selected features
selected_features = ifs.drop_selected_cols(df, is_all_true=True)
print("Selected features:", selected_features.columns.tolist())
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Masashi Ueda