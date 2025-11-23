import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

TRAIN_PATH = r"...SCRM_timeSeries_2018_train.csv"
TEST_PATH  = r"...SCRM_timeSeries_2018_test.csv"
STRATIFY_COL = 'SCMstability_category'
RANDOM_STATE = 42
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.8, 0.1, 0.1   # 8:1:1

def process_and_split_data(train_path: str, test_path: str, stratify_col: str):
    # 1) Reading and Basic Checks
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    train_df.name, test_df.name = 'Train(Original)', 'Test(Original)'
    print(f"[LOAD] Training set: {train_df.shape}, Test set: {test_df.shape}")

    if stratify_col not in train_df.columns or stratify_col not in test_df.columns:
        raise ValueError(f"The label column '{stratify_col}' is missing. Please check both files.")

    # 2) Merge and deduplicate (only rows with exact duplicates will be removed)
    df_all = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"[MERGE] After merging and removing duplicates: {df_all.shape}")

    # 3) Missing Value Handling
    cols_first5 = df_all.columns[:5]
    nan_cnt = df_all[cols_first5].isna().sum().sum()
    if nan_cnt > 0:
        df_all[cols_first5] = df_all[cols_first5].fillna(0)
        print(f"[IMPUTE] The first five columns are empty. 0：{nan_cnt} ")
    else:
        print("[IMPUTE] The first five columns contain no empty values.")

    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    if stratify_col in numeric_cols:
        numeric_cols.remove(stratify_col)
    nan_cnt_all = df_all[numeric_cols].isna().sum().sum()
    if nan_cnt_all > 0:
        df_all[numeric_cols] = df_all[numeric_cols].fillna(0)
        print(f"[IMPUTE] Fill additional null values for numerical features 0：{nan_cnt_all} ")
    else:
        print("[IMPUTE] Numeric features contain no null values")

    # 4) Hierarchical division 8:1:1
    X_all = df_all.drop(columns=[stratify_col])
    y_all = df_all[stratify_col].astype('int64').values

    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X_all, y_all,
        test_size=SPLIT_TEST,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    val_ratio_in_tmp = SPLIT_VAL / (SPLIT_TRAIN + SPLIT_VAL)  # 0.1 / 0.9 = 0.111...
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp,
        test_size=val_ratio_in_tmp,
        random_state=RANDOM_STATE,
        stratify=y_tmp
    )

    print(f"[SPLIT] New training set: {X_tr.shape}, validation set: {X_va.shape}, test set: {X_te.shape}")

    def ratio(y):
        v = pd.Series(y).value_counts(normalize=True).sort_index()
        return (v*100).round(2)

    print("\n[CLASS RATIO %]")
    print("Train:\n", ratio(y_tr))
    print("Val  :\n", ratio(y_va))
    print("Test :\n", ratio(y_te))

    # 5) Standardization
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_te)
    print("\n[SCALE]  StandardScaler completed.")

    # 6) Reassemble into a DataFrame and save
    feature_cols = X_all.columns.tolist()
    out_dir = Path(train_path).parent

    train_out = pd.DataFrame(X_tr_scaled, columns=feature_cols)
    train_out[STRATIFY_COL] = y_tr
    val_out = pd.DataFrame(X_va_scaled, columns=feature_cols)
    val_out[STRATIFY_COL] = y_va
    test_out = pd.DataFrame(X_te_scaled, columns=feature_cols)
    test_out[STRATIFY_COL] = y_te

    train_path_new = out_dir / "SCRM_train_mix.csv"
    val_path_new   = out_dir / "SCRM_val_mix.csv"
    test_path_new  = out_dir / "SCRM_test_mix.csv"

    train_out.to_csv(train_path_new, index=False)
    val_out.to_csv(val_path_new, index=False)
    test_out.to_csv(test_path_new, index=False)

    # 7) Save Standardizer
    scaler_path = out_dir / "standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print("\n[SAVE] New dataset saved：")
    print(f"- Training set: {train_path_new}")
    print(f"- Validation set: {val_path_new}")
    print(f"- Test set: {test_path_new}")
    print(f"- Scaler: {scaler_path}")

if __name__ == "__main__":
    process_and_split_data(TRAIN_PATH, TEST_PATH, STRATIFY_COL)
