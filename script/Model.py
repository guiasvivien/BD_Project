import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path, target_column, excluded_features):
    df = pd.read_csv(file_path)

    # Encode the target variable
    lb = LabelEncoder()
    df[target_column] = lb.fit_transform(df[target_column])

    # Keep only numerical columns
    numeric_df = df.select_dtypes(include=[int, float])
    numeric_df[target_column] = df[target_column]
    numeric_df = numeric_df.drop(columns=excluded_features, errors='ignore')

    # Fill NaN values with column means
    filled_df = numeric_df.fillna(numeric_df.mean())

    # Separate features and target
    y = filled_df[target_column]
    X = filled_df.drop(columns=[target_column], errors='ignore')

    return X, y, lb

# Split data into training and testing sets
def split_data(X, y, test_size=0.15, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Apply SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Feature selection using Random Forest
def select_features(X_train, y_train, threshold=0.04):
    rf_selector = RandomForestClassifier(random_state=42)
    rf_selector.fit(X_train, y_train)
    importances = rf_selector.feature_importances_
    important_features = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    selected_features = important_features[important_features['Importance'] >= threshold]['Feature']
    return selected_features

# Build and train Neural Network
def build_and_train_nn(X_train, y_train, X_test, y_test):
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)

    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dropout(0.3),
        Dense(y_train_categorical.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(X_train, y_train_categorical, epochs=100, batch_size=16, verbose=1)
    eval_nn = model.evaluate(X_test, y_test_categorical, verbose=1)
    print("Neural Network Accuracy:", eval_nn[1])

# Main workflow
if __name__ == "__main__":
    # Configuration
    file_path = r"D:\\Faculty\\Master_2\\Big_data\\BD_project\\data\\data_1_2.csv"
    target_column = 'sleepScores_overall_qualifierKey'
    excluded_features = ['sleepScores_overall']

    # Load and preprocess data
    X, y, label_encoder = load_and_preprocess_data(file_path, target_column, excluded_features)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

        # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

        # Scale features
    scaler = StandardScaler()
    X_train_smote = scaler.fit_transform(X_train_smote)
    X_test = scaler.transform(X_test)

        # Logistic Regression with hyperparameter tuning
    lr = LogisticRegression(max_iter=1000000000, class_weight="balanced")
    param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100, 200], 'solver': ['liblinear', 'lbfgs']}
    lr_best = hyperparameter_tuning(lr, param_grid_lr, X_train_smote, y_train_smote)
    train_and_evaluate_model(lr_best, X_train_smote, y_train_smote, X_test, y_test, "Logistic Regression")

        # Random Forest with hyperparameter tuning
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    rf_best = hyperparameter_tuning(rf, param_grid_rf, X_train_smote, y_train_smote)
    train_and_evaluate_model(rf_best, X_train_smote, y_train_smote, X_test, y_test, "Random Forest")

        # XGBoost with hyperparameter tuning
    xgb_model = xgb.XGBClassifier(random_state=42)
    param_grid_xgb = {'learning_rate': [0.001, 0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200, 250],
                          'max_depth': [3, 5, 7, 10]}
    xgb_best = hyperparameter_tuning(xgb_model, param_grid_xgb, X_train_smote, y_train_smote)
    train_and_evaluate_model(xgb_best, X_train_smote, y_train_smote, X_test, y_test, "XGBoost")

        # Feature selection
    selected_features = select_features(pd.DataFrame(X_train_smote, columns=X.columns), y_train_smote)
    X_train_selected = pd.DataFrame(X_train_smote, columns=X.columns)[selected_features]
    X_test_selected = pd.DataFrame(X_test, columns=X.columns)[selected_features]

        # Train and evaluate Neural Network
    build_and_train_nn(X_train_selected, y_train_smote, X_test_selected, y_test)

