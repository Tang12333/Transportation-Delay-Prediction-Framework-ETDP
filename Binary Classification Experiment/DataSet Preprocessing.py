import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from datetime import datetime
from geopy.distance import geodesic as geopy_haversine
GEOPY_AVAILABLE = True
import warnings

warnings.filterwarnings('ignore')


class SupplyChainDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.label_encoders = {}

    def load_and_preprocess_raw_data(self, raw_data_path):
        """
        Load and preprocess from the raw dataset
        """
        print("Load the original dataset...")
        df = pd.read_csv(raw_data_path, encoding='utf-8')
        print(f"Raw data shape: {df.shape}")

        df_processed = self._basic_preprocessing(df)
        print(f"Preliminary preprocessing completed, shape: {df_processed.shape}")

        # Feature Engineering
        df_processed = self._advanced_feature_engineering(df_processed)
        print(f"Feature engineering completed, shape: {df_processed.shape}")

        # Category Feature Encoding
        df_processed = self._encode_categorical_features(df_processed)
        print(f"Category feature encoding completed, shape: {df_processed.shape}")

        return df_processed

    def _basic_preprocessing(self, df):
        """
        Data Cleaning and Preprocessing
        """
        # Remove redundant features
        useless_features = [
            'customer_email', 'customer_fname', 'customer_lname',
            'customer_password','order_state'
            'order_customer_id', 'order_item_id','order_city'
            'product_image', 'address_dest','order_country'
        ]
        df = df.drop(columns=[col for col in useless_features if col in df.columns])

        df = df.fillna(0)  # Fill missing values

        return df

    def _advanced_feature_engineering(self, df):
        """
        Feature Engineering
        """
        print("Begin feature engineering...")

        # 1. Time-related characteristics
        df['order_date'] = pd.to_datetime(df['order_date_dateorders'])
        df['shipping_date'] = pd.to_datetime(df['shipping_date_dateorders'])

        # Order Processing Time
        df['order_processing_days'] = (df['shipping_date'] - df['order_date']).dt.days
        print("Constructing Order Processing Time Bin Features...")
        # Custom Partitioning Based on Data Distribution and Business Logic
        bins_proc_days = [-1, 0, 2, 4, 6, np.inf] # (0], (0,2], (2,4], (4,6], (6, inf]
        labels_proc_days = ['0_days', '1_2_days', '3_4_days', '5_6_days', '7_plus_days']
        df['order_processing_days_binned'] = pd.cut(df['order_processing_days'], bins=bins_proc_days, labels=labels_proc_days, right=True)
        print(f"Order processing time binning feature construction completed. New features added.: 'order_processing_days_binned'")
        print("Category Distribution of Order Processing Times：")
        print(df['order_processing_days'].value_counts())
        # Order Month, Quarter, Week
        df['order_month'] = df['order_date'].dt.month
        df['order_quarter'] = df['order_date'].dt.quarter
        df['order_weekday'] = df['order_date'].dt.weekday

        # Is it an order placed over the weekend?
        df['is_weekend_order'] = (df['order_weekday'] >= 5).astype(int)

        # 2. Delivery-related characteristics
        df['shipping_delay_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']

        if 'late_delivery_risk' in df.columns:
            df['delivery_risk'] = df['late_delivery_risk'].astype(int)
        else:
            df['delivery_risk'] = (df['shipping_delay_days'] > 0).astype(int)

        # Delivery on time or not
        df['on_time_delivery'] = (df['shipping_delay_days'] <= 0).astype(int)

        # Delivery Efficiency Ratio
        df['delivery_efficiency_ratio'] = np.where(
            df['days_for_shipment_scheduled'] > 0,
            df['days_for_shipping_real'] / df['days_for_shipment_scheduled'],
            0
        )

        # 3. Financial-related characteristics
        # Discount amount as a percentage of the original price
        df['discount_percentage'] = np.where(
            df['order_item_product_price'] > 0,
            df['order_item_discount'] / df['order_item_product_price'],
            0
        )

        # Actual payment amount
        df['actual_payment'] = df['order_item_product_price'] - df['order_item_discount']

        # Profit margin
        df['profit_margin'] = np.where(
            df['actual_payment'] > 0,
            df['order_profit_per_order'] / df['actual_payment'],
            0
        )

        # 4. Geographical Features
        if GEOPY_AVAILABLE:
            print("Using geopy to calculate the Haversine distance...")

            def calculate_haversine(row):
                try:
                    src = (row['latitude_src'], row['longitude_src'])
                    dest = (row['latitude_dest'], row['longitude_dest'])
                    return geopy_haversine(src, dest).kilometers
                except Exception:
                    return np.nan

            df['haversine_distance'] = df.apply(calculate_haversine, axis=1)
            df['geo_distance_approx'] = abs(df['latitude_dest'] - df['latitude_src']) + abs(
                df['longitude_dest'] - df['longitude_src'])
            df['haversine_distance'].fillna(df['geo_distance_approx'], inplace=True)
            df.drop(columns=['geo_distance_approx'], inplace=True, errors='ignore')
        else:
            print("geopy is unavailable; using geo_distance_approx with Manhattan distance approximation...")
            df['haversine_distance'] = abs(df['latitude_dest'] - df['latitude_src']) + abs(
                df['longitude_dest'] - df['longitude_src'])

        # 5. Product-Related Features
        # Product Price Range
        df['price_range'] = pd.cut(df['product_price'],
                                   bins=[0, 50, 100, 200, 500, np.inf],
                                   labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])

        # 6. Order Complexity Features
        if 'order_item_cardprod_id' in df.columns:
            df['order_num_distinct_items'] = df.groupby('order_id')['order_item_cardprod_id'].transform('nunique')
        else:
            df['order_num_distinct_items'] = 1

        df['is_shipping_date_holiday'] = 0

        # Delete id columns that are no longer needed
        id_columns_to_drop = ['customer_id', 'product_card_id','order_id']
        df = df.drop(columns=[col for col in id_columns_to_drop if col in df.columns])

        # Delete other unnecessary columns
        df = df.drop(columns=['order_date_dateorders', 'shipping_date_dateorders', 'order_date', 'shipping_date','shipping_delay_days',
                             'delivery_efficiency_ratio', 'days_for_shipping_real','days_for_shipment_scheduled','late_delivery_risk'])

        return df

    def _encode_categorical_features(self, df):
        """
        Category Feature Encoding
        """
        print("Start Category Feature Encoding...")

        # Medium-cardinality category attributes (using tag encoding)
        medium_cardinality_features = [
            'category_name', 'customer_segment', 'department_name',
            'market', 'order_region', 'order_status', 'shipping_mode',
            'order_city_en', 'order_state_en', 'order_country_en',
            'customer_city', 'customer_country',
             'product_name'
        ]

        # Low-Base Category Features (Using Unigram Encoding)
        low_cardinality_features = [
            'type', 'delivery_status', 'on_time_delivery', 'is_weekend_order',
            'price_range','order_processing_days_binned'
        ]

        # Handling Medium-Cardinality Features (Label Encoding)
        for feature in medium_cardinality_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le

        if 'is_shipping_date_holiday' in df.columns:

        df = pd.get_dummies(df, columns=low_cardinality_features, dummy_na=True)

        return df

    def select_features(self, df, target_column='delivery_risk'):
        """
        Feature Selection
        """
        print("Start feature selection...")

        # Separate features and labels
        target_columns = ['delivery_risk']
        targets = df[target_columns]
        features = df.drop(columns=target_columns)

        print(f"Number of features: {len(features.columns)}")
        print(f"Distribution of Feature Types:")
        print(features.dtypes.value_counts())

        numeric_features = features.select_dtypes(include=[np.number]).columns
        non_numeric_features = features.select_dtypes(exclude=[np.number]).columns
        print(f"Numerical characteristics: {len(numeric_features)}")
        print(f"Non-numeric characteristics: {len(non_numeric_features)}")
        if len(non_numeric_features) > 0:
            print("Non-numeric characteristics:", non_numeric_features.tolist())

        X_numeric = features[numeric_features]

        # 1. Variance Filtering
        from sklearn.feature_selection import VarianceThreshold
        selector_var = VarianceThreshold(threshold=0.01)
        try:
            X_var_filtered = selector_var.fit_transform(X_numeric)
            selected_features_var = X_numeric.columns[selector_var.get_support()]
            print(f"Features retained after variance filtering: {len(selected_features_var)}")

            # 2. Statistical Testing Feature Selection
            y = targets[target_column]
            selector_stats = SelectKBest(score_func=f_classif, k=min(50, len(selected_features_var)))
            X_selected = selector_stats.fit_transform(X_var_filtered, y)

            self.feature_selector = selector_stats
            self.selected_features = selected_features_var[selector_stats.get_support()]
            print(f"Final Feature Selection: {len(self.selected_features)}")
            print("Selected Key Features:")
            for i, feature in enumerate(self.selected_features[:20]):
                print(f"  {i + 1}. {feature}")

            X_final = pd.DataFrame(X_selected, columns=self.selected_features)
            return pd.concat([X_final, targets.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(f"Error occurred during feature selection: {e}")
            return pd.concat([X_numeric, targets.reset_index(drop=True)], axis=1)

    def standardize_features(self, df, is_training=True):
        """
        Standardization
        """
        target_columns = ['delivery_risk']
        targets = df[target_columns] if all(col in df.columns for col in target_columns) else None
        features = df.drop(columns=target_columns) if targets is not None else df

        if is_training:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

        if targets is not None:
            result = pd.concat([features_scaled_df, targets.reset_index(drop=True)], axis=1)
            return result
        else:
            return features_scaled_df


def create_train_val_test_split(df, test_size=0.1, val_size=0.1, random_state=42):
    """
    Create training, validation, and test sets
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['delivery_risk']
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_val_df['delivery_risk']
    )

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    print("\nLabel Distribution:")
    print("Training Set:", train_df['delivery_risk'].value_counts().to_dict())
    print("Test Set:", test_df['delivery_risk'].value_counts().to_dict())

    return train_df, val_df, test_df


def analyze_data_quality(df, target_column='delivery_risk'):
    """
    Analyze data quality
    """
    print(f"Data Shape: {df.shape}")

    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("Missing Value Statistics:")
        print(missing_data[missing_data > 0])
    else:
        print("No missing values")

    if target_column in df.columns:
        print(f"\n{target_column} Distribution:")
        print(df[target_column].value_counts())
        print(f"Category Proportion: {df[target_column].value_counts(normalize=True).to_dict()}")

    # delivery risk distribution
    if 'delivery_risk' in df.columns:
        print(f"\ndelivery risk distribution:")
        print(df['delivery_risk'].value_counts())
        print(f"Category Proportion: {df['delivery_risk'].value_counts(normalize=True).to_dict()}")

    # Feature Validity Analysis
    if target_column in df.columns:
        feature_columns = [col for col in df.columns if col not in ['delivery_risk']]
        X = df[feature_columns]
        y = df[target_column]

        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]

        if len(X_numeric.columns) > 0:
            try:
                from sklearn.feature_selection import f_classif
                f_scores, p_values = f_classif(X_numeric, y)
                valid_features = sum(p_values < 0.05)
                weak_features = sum(p_values > 0.01)
                print(f"\nFeature Validity Analysis ({target_column}):")
                print(f"Number of effective features (p<0.05): {valid_features} / {len(X_numeric.columns)}")
                print(f"Number of weak-effect features (p>0.01): {weak_features} / {len(X_numeric.columns)}")
                return valid_features, weak_features
            except Exception as e:
                print(f"Feature validity analysis error: {e}")
                return 0, 0
        else:
            print("There are no numerical characteristics available for analysis.")
            return 0, 0


def analyze_feature_importance(df):
    """
    Analyzing Feature Importance
    """
    from sklearn.ensemble import RandomForestClassifier

    feature_columns = [col for col in df.columns if col not in ['delivery_risk']]
    X = df[feature_columns]
    y_binary = df['delivery_risk']

    numeric_features = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_features]
    print(f"Number of features for feature importance analysis: {len(X_numeric.columns)}")

    print("\nFeature Importance in Binary Classification Tasks:")
    rf_binary = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_binary.fit(X_numeric, y_binary)
    feature_importance_binary = pd.DataFrame({
        'feature': X_numeric.columns,
        'importance': rf_binary.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance_binary.head(15))

    return feature_importance_binary


def main():
    raw_data_path = r".../DataCoSupplyChainDatasetRefined.csv"
    preprocessor = SupplyChainDataPreprocessor()

    try:
        # 1. Load and preprocess raw data
        print("=" * 50)
        print("Load and preprocess raw data")
        print("=" * 50)
        df_processed = preprocessor.load_and_preprocess_raw_data(raw_data_path)

        # 2. Data Quality Analysis
        print("\n" + "=" * 50)
        print("Data Quality Analysis")
        print("=" * 50)
        analyze_data_quality(df_processed, 'delivery_risk')

        # 3. Feature Selection
        print("\n" + "=" * 50)
        print("Feature Selection")
        print("=" * 50)
        df_selected = preprocessor.select_features(df_processed, target_column='delivery_risk')

        # 4. Dataset Partitioning
        print("\n" + "=" * 50)
        print("Dataset Partitioning")
        print("=" * 50)
        train_df, val_df, test_df = create_train_val_test_split(df_selected)

        # 5. Feature Standardization
        print("\n" + "=" * 50)
        print("Feature Standardization")
        print("=" * 50)
        train_scaled = preprocessor.standardize_features(train_df, is_training=True)
        val_scaled = preprocessor.standardize_features(val_df, is_training=False)
        test_scaled = preprocessor.standardize_features(test_df, is_training=False)

        # 6. Save the processed data
        print("\n" + "=" * 50)
        print("Save the processed data")
        print("=" * 50)
        output_path = r"..."
        train_scaled.to_csv(f"{output_path}/train_scaled_binary.csv", index=False)
        val_scaled.to_csv(f"{output_path}/val_scaled_binary.csv", index=False)
        test_scaled.to_csv(f"{output_path}/test_scaled_binary.csv", index=False)
        print(f"Training set shape: {train_scaled.shape}")
        print(f"Validation set shape: {val_scaled.shape}")
        print(f"Test set shape: {test_scaled.shape}")

        # 7. Feature Importance Analysis
        print("\n" + "=" * 50)
        print("Feature Importance Analysis")
        print("=" * 50)
        analyze_feature_importance(train_scaled)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()