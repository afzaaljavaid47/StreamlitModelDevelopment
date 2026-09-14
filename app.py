import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Trading Analyzer",
    page_icon="📈",
    layout="wide"
)

# st.title("📈 AI Trading Analyzer")
# st.write(
#     "Upload your data, choose Classification or Regression, "
#     "configure a machine learning model, train it, save it, "
#     "and test it on new data."
# )


# =========================================================
# DIRECTORIES
# =========================================================

MODEL_DIR = "saved_models"
# MODEL_PATH = os.path.join(MODEL_DIR, "trading_model.pkl")
# INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
#
# os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# SESSION STATE
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def make_one_hot_encoder():
    """
    Supports newer and older scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )
def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                make_one_hot_encoder(),
                categorical_features
            )
        )

    if not transformers:
        raise ValueError(
            "No usable features were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features

def save_model(model, model_info,model_name):
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    info_path = os.path.join(model_dir, "model_info.json")
    joblib.dump(model, model_path)
    with open(info_path, "w") as f:
        json.dump(
            model_info,
            f,
            indent=4
        )
def get_saved_models():
    if not os.path.exists(MODEL_DIR):
        return []
    model_names = []
    for name in os.listdir(MODEL_DIR):
        path = os.path.join(MODEL_DIR, name)
        if os.path.isdir(path):
            model_names.append(name)
    return model_names
@st.cache_resource
def load_saved_model(name):
    if not os.path.exists(MODEL_DIR):
        return None, None
    model_dir = os.path.join(MODEL_DIR, name)

    model_path = os.path.join(model_dir, "model.pkl")
    info_path = os.path.join(model_dir, "model_info.json")

    if not os.path.exists(model_path):
        return None, None

    if not os.path.exists(info_path):
        return None, None

    model = joblib.load(model_path)

    with open(info_path, "r") as f:
        model_info = json.load(f)

    return model, model_info
# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

page = st.sidebar.radio(
    "Choose Section",
    [
        "Train New Model",
        "Test Trained Model"
    ]
)
# =========================================================
# TRAIN NEW MODEL
# =========================================================
if page == "Train New Model":
    st.header("🧠 Train New Model")
    col1, col2 = st.columns(2)

    with col1:
        model_name_reference = st.text_input(
            "Model Name",
            placeholder="Enter model name..."
        )

    with col2:
        train_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"]
        )

    if train_file is not None:
        try:
            df = pd.read_csv(train_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            st.stop()
        st.success(
            f"Training data loaded successfully: "
            f"{df.shape[0]} rows × {df.shape[1]} columns"
        )
        with st.expander("👀 Preview Training Data"):
            st.dataframe(
                df.head(20),
                width="stretch"
            )
        # -------------------------------------------------
        # Dataset information
        # -------------------------------------------------
        st.subheader("📊 Dataset Information")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )
        col4.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        # -------------------------------------------------
        # 2. Problem type
        # -------------------------------------------------

        st.subheader("2️⃣ Select Problem Type")

        problem_type = st.radio(
            "What type of prediction do you want?",
            [
                "Classification",
                "Regression"
            ],
            horizontal=True,
            help=(
                "Classification predicts categories such as "
                "UP/DOWN or BUY/SELL. Regression predicts "
                "continuous numbers such as house prices."
            )
        )

        if problem_type == "Classification":
            st.info(
                "📌 Classification example: predict UP/DOWN, "
                "BUY/SELL, or 0/1."
            )
        else:
            st.info(
                "📌 Regression example: predict a continuous "
                "value such as price."
            )

        # -------------------------------------------------
        # 3. Select target
        # -------------------------------------------------

        st.subheader("3️⃣ Select Target Column")

        target_column = st.selectbox(
            "What do you want the model to predict?",
            df.columns
        )

        # -------------------------------------------------
        # Target preview / validation
        # -------------------------------------------------

        target_preview = df[target_column].dropna()

        if target_preview.empty:
            st.error(
                "The selected target column contains no valid values."
            )
            st.stop()

        unique_target_count = target_preview.nunique()

        col1, col2 = st.columns(2)

        col1.metric(
            "Target Values",
            len(target_preview)
        )

        col2.metric(
            "Unique Target Values",
            unique_target_count
        )

        if problem_type == "Classification":

            if unique_target_count < 2:
                st.error(
                    "Classification requires at least 2 different classes."
                )
                st.stop()

            class_counts = target_preview.value_counts()

            st.write("**Class distribution:**")
            st.dataframe(
                class_counts.rename("Count").to_frame(),
                width="stretch"
            )

            if class_counts.min() < 2:
                st.warning(
                    "⚠️ At least one class has only one row. "
                    "Stratified train/test splitting requires "
                    "at least 2 samples per class."
                )

        else:

            if not pd.api.types.is_numeric_dtype(target_preview):
                st.error(
                    "Regression target must be numeric. "
                    "Please select a numeric target column."
                )
                st.stop()

        # -------------------------------------------------
        # 4. Select features
        # -------------------------------------------------

        st.subheader("4️⃣ Select Features")

        available_features = [
            column for column in df.columns
            if column != target_column
        ]

        selected_features = st.multiselect(
            "Select columns that the model should use",
            available_features,
            default=available_features
        )

        if not selected_features:
            st.warning(
                "Please select at least one feature."
            )
            st.stop()

        # -------------------------------------------------
        # 5. Training settings
        # -------------------------------------------------

        st.subheader("5️⃣ Training Settings")

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Validation/Test Size",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05
            )

        with col2:
            random_state = st.number_input(
                "Random State",
                min_value=0,
                value=42,
                step=1
            )

        # -------------------------------------------------
        # 6. Select model
        # -------------------------------------------------

        st.subheader("6️⃣ Select Machine Learning Model")

        if problem_type == "Classification":

            model_name = st.selectbox(
                "Choose Classification Model",
                [
                    "Random Forest",
                    "Logistic Regression"
                ]
            )

        else:

            model_name = st.selectbox(
                "Choose Regression Model",
                [
                    "Random Forest",
                    "Linear Regression"
                ]
            )

        # =================================================
        # RANDOM FOREST SETTINGS
        # =================================================

        if model_name == "Random Forest":

            st.subheader("🌲 Random Forest Settings")

            col1, col2 = st.columns(2)

            with col1:

                n_estimators = st.slider(
                    "Number of Trees",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )

                max_depth_option = st.selectbox(
                    "Maximum Tree Depth",
                    [
                        "None",
                        5,
                        10,
                        20,
                        30,
                        50
                    ]
                )

                if max_depth_option == "None":
                    max_depth = None
                else:
                    max_depth = int(max_depth_option)

                min_samples_split = st.slider(
                    "Minimum Samples to Split",
                    min_value=2,
                    max_value=20,
                    value=2
                )

            with col2:

                min_samples_leaf = st.slider(
                    "Minimum Samples per Leaf",
                    min_value=1,
                    max_value=20,
                    value=1
                )

                max_features = st.selectbox(
                    "Maximum Features",
                    [
                        "sqrt",
                        "log2",
                        None
                    ]
                )

                if problem_type == "Classification":

                    class_weight = st.selectbox(
                        "Class Weight",
                        [
                            None,
                            "balanced",
                            "balanced_subsample"
                        ]
                    )

                else:
                    class_weight = None

            model_settings = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "class_weight": class_weight
            }

        # =================================================
        # LOGISTIC REGRESSION SETTINGS
        # =================================================

        elif model_name == "Logistic Regression":

            st.subheader("📈 Logistic Regression Settings")

            col1, col2 = st.columns(2)

            with col1:

                C = st.number_input(
                    "C (Regularization)",
                    min_value=0.001,
                    max_value=100.0,
                    value=1.0,
                    step=0.1
                )

                max_iter = st.number_input(
                    "Maximum Iterations",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100
                )

            with col2:

                solver = st.selectbox(
                    "Solver",
                    [
                        "lbfgs",
                        "liblinear",
                        "newton-cg",
                        "newton-cholesky",
                        "sag",
                        "saga"
                    ]
                )

                class_weight = st.selectbox(
                    "Class Weight",
                    [
                        None,
                        "balanced"
                    ]
                )

            model_settings = {
                "C": C,
                "max_iter": max_iter,
                "solver": solver,
                "class_weight": class_weight
            }

        # =================================================
        # LINEAR REGRESSION SETTINGS
        # =================================================

        else:

            st.subheader("📉 Linear Regression Settings")

            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Allow the model to learn the intercept."
            )

            model_settings = {
                "fit_intercept": fit_intercept
            }

        # -------------------------------------------------
        # 7. Train button
        # -------------------------------------------------

        st.divider()

        train_button = st.button(
            "🚀 Train Model",
            type="primary",
            width="stretch"
        )

        if train_button:

            # ---------------------------------------------
            # Prepare X and y
            # ---------------------------------------------

            X = df[selected_features].copy()
            y = df[target_column].copy()

            # ---------------------------------------------
            # Remove rows with missing target
            # ---------------------------------------------

            valid_rows = y.notna()

            X = X.loc[valid_rows].copy()
            y = y.loc[valid_rows].copy()

            if len(X) < 2:
                st.error(
                    "Not enough valid rows to train the model."
                )
                st.stop()

            # ---------------------------------------------
            # Clean feature columns
            # ---------------------------------------------

            # Drop rows where all selected features are missing.
            feature_valid_rows = ~X.isna().all(axis=1)

            X = X.loc[feature_valid_rows].copy()
            y = y.loc[feature_valid_rows].copy()

            if len(X) < 2:
                st.error(
                    "Not enough usable feature rows after cleaning."
                )
                st.stop()

            # ---------------------------------------------
            # Regression target conversion
            # ---------------------------------------------

            if problem_type == "Regression":

                y = pd.to_numeric(
                    y,
                    errors="coerce"
                )

                valid_numeric_target = y.notna()

                X = X.loc[valid_numeric_target].copy()
                y = y.loc[valid_numeric_target].copy()

                if len(X) < 2:
                    st.error(
                        "The regression target does not contain "
                        "enough numeric values."
                    )
                    st.stop()

            # ---------------------------------------------
            # Detect numerical/categorical columns
            # ---------------------------------------------

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            # ---------------------------------------------
            # Create preprocessor
            # ---------------------------------------------

            try:

                preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            StandardScaler(),
                            numeric_features
                        )
                    ] if numeric_features else [],
                    remainder="passthrough"
                )

                # Rebuild properly when categorical columns exist.
                transformers = []

                if numeric_features:
                    transformers.append(
                        (
                            "numeric",
                            StandardScaler(),
                            numeric_features
                        )
                    )

                if categorical_features:
                    transformers.append(
                        (
                            "categorical",
                            make_one_hot_encoder(),
                            categorical_features
                        )
                    )

                if not transformers:
                    st.error(
                        "No usable feature columns were found."
                    )
                    st.stop()

                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder="drop"
                )

            except Exception as e:
                st.error(
                    f"Could not create preprocessing pipeline: {e}"
                )
                st.stop()

            # ---------------------------------------------
            # Classification validation
            # ---------------------------------------------

            stratify_value = None

            if problem_type == "Classification":

                class_counts = y.value_counts()

                if len(class_counts) < 2:
                    st.error(
                        "Classification requires at least 2 classes."
                    )
                    st.stop()

                # Use stratification only when every class has
                # at least 2 samples.
                if class_counts.min() >= 2:

                    # Also ensure the validation set can contain
                    # at least one sample from every class.
                    validation_rows = int(
                        np.ceil(len(y) * test_size)
                    )

                    if validation_rows >= len(class_counts):
                        stratify_value = y
                    else:
                        st.warning(
                            "The validation set is too small to "
                            "contain every class, so stratification "
                            "has been disabled."
                        )

                else:
                    st.warning(
                        "Some classes contain only one sample. "
                        "Stratification has been disabled."
                    )

            # ---------------------------------------------
            # Train/Test split
            # ---------------------------------------------

            try:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_value
                )

            except Exception as e:
                st.error(
                    f"Could not split the dataset: {e}"
                )
                st.stop()

            # ---------------------------------------------
            # Create model
            # ---------------------------------------------

            if problem_type == "Classification":

                if model_name == "Random Forest":

                    estimator = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=-1
                    )

                else:

                    # Some solvers do not support every combination.
                    if (
                        solver in ["lbfgs", "newton-cg", "newton-cholesky", "sag"]
                        and class_weight not in [None, "balanced"]
                    ):
                        class_weight = None

                    estimator = LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver=solver,
                        class_weight=class_weight,
                        random_state=random_state
                    )

            else:

                if model_name == "Random Forest":

                    estimator = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        n_jobs=-1
                    )

                else:

                    estimator = LinearRegression(
                        fit_intercept=fit_intercept
                    )

            # ---------------------------------------------
            # Create pipeline
            # ---------------------------------------------

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator)
                ]
            )

            # ---------------------------------------------
            # Train
            # ---------------------------------------------

            st.info(
                f"🔄 Training {model_name} "
                f"({problem_type}) on {len(X_train):,} rows..."
            )

            progress = st.progress(0)

            try:

                progress.progress(10)

                with st.spinner(
                    "Training model... Please wait."
                ):

                    pipeline.fit(
                        X_train,
                        y_train
                    )

                progress.progress(100)

            except Exception as e:

                progress.empty()

                st.error(
                    "❌ Training failed."
                )

                st.exception(e)
                st.stop()

            progress.empty()

            st.success(
                "✅ Model trained successfully!"
            )

            # ---------------------------------------------
            # Validation predictions
            # ---------------------------------------------

            try:

                y_pred = pipeline.predict(
                    X_test
                )

            except Exception as e:

                st.error(
                    f"Prediction failed after training: {e}"
                )
                st.stop()

            # =================================================
            # CLASSIFICATION RESULTS
            # =================================================

            if problem_type == "Classification":

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                model_classes = [
                    str(x)
                    for x in pipeline.named_steps["model"].classes_
                ]

                # ---------------------------------------------
                # Save model information
                # ---------------------------------------------

                model_info = {

                    "problem_type": problem_type,

                    "model_name": model_name,

                    "target_column": target_column,

                    "features": selected_features,

                    "test_size": float(test_size),

                    "random_state": int(random_state),

                    "accuracy": float(accuracy),

                    "model_settings": model_settings,

                    "training_rows": int(len(X_train)),

                    "validation_rows": int(len(X_test)),

                    "classes": model_classes
                }

                save_model(
                    pipeline,
                    model_info,
                    model_name_reference
                )

                st.session_state.trained_models[model_name_reference] = {
                    "model":pipeline,
                    "model_info":model_info
                }
                # ---------------------------------------------
                # Show results
                # ---------------------------------------------

                st.subheader("📊 Model Performance")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Validation Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                col2.metric(
                    "Training Rows",
                    len(X_train)
                )

                col3.metric(
                    "Validation Rows",
                    len(X_test)
                )

                st.subheader("📋 Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df,
                    width="stretch"
                )

                st.subheader("🔢 Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=model_classes[:len(cm)],
                    columns=model_classes[:len(cm)]
                )

                st.dataframe(
                    cm_df,
                    width="stretch"
                )

            # =================================================
            # REGRESSION RESULTS
            # =================================================

            else:

                mae = mean_absolute_error(
                    y_test,
                    y_pred
                )

                mse = mean_squared_error(
                    y_test,
                    y_pred
                )

                rmse = np.sqrt(mse)

                r2 = r2_score(
                    y_test,
                    y_pred
                )

                model_info = {

                    "problem_type": problem_type,

                    "model_name": model_name,

                    "target_column": target_column,

                    "features": selected_features,

                    "test_size": float(test_size),

                    "random_state": int(random_state),

                    "mae": float(mae),

                    "mse": float(mse),

                    "rmse": float(rmse),

                    "r2": float(r2),

                    "model_settings": model_settings,

                    "training_rows": int(len(X_train)),

                    "validation_rows": int(len(X_test))
                }

                save_model(
                    pipeline,
                    model_info,
                    model_name_reference
                )

                st.session_state.trained_models[model_name_reference] = {
                    "model": pipeline,
                    "model_info": model_info
                }

                # ---------------------------------------------
                # Show results
                # ---------------------------------------------

                st.subheader("📊 Regression Performance")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "MAE",
                    f"{mae:,.2f}"
                )

                col2.metric(
                    "RMSE",
                    f"{rmse:,.2f}"
                )

                col3.metric(
                    "R² Score",
                    f"{r2:.4f}"
                )

                col4.metric(
                    "R² Percentage",
                    f"{r2 * 100:.2f}%"
                )

                st.info(
                    "💡 For regression, R² is not exactly the same "
                    "thing as classification accuracy. Use MAE, "
                    "RMSE and R² together to judge the model."
                )

            # ---------------------------------------------
            # Saved confirmation
            # ---------------------------------------------

            st.info(
                "💾 Model and model settings have been saved "
                "inside the saved_models folder."
            )
# =========================================================
# TEST MODEL
# =========================================================
elif page == "Test Trained Model":

    st.header("🧪 Test Trained Model")

    # -----------------------------------------------------
    # Get All models
    # -----------------------------------------------------
    models=get_saved_models()
    # select_model=st.selectbox("Select Model",models)
    # -----------------------------------------------------
    select_model = st.selectbox(
        "Select Model",
        models,
        index=None,
        placeholder="Select Model"
    )

    if select_model is None:
        st.stop()

    # Only runs after user selects a model
    st.write("Selected model:", select_model)

    model,model_info=load_saved_model(select_model)
    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    problem_type = model_info.get(
        "problem_type",
        "Classification"
    )

    st.success(
        f"Loaded model: {model_info['model_name']} "
        f"({problem_type})"
    )

    # -----------------------------------------------------
    # Show model information
    # -----------------------------------------------------

    st.subheader("ℹ️ Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model",
        model_info["model_name"]
    )

    col2.metric(
        "Problem Type",
        problem_type
    )

    col3.metric(
        "Features",
        len(model_info["features"])
    )

    st.write(
        "**Features:**",
        ", ".join(model_info["features"])
    )

    st.write(
        "**Target:**",
        model_info["target_column"]
    )

    # -----------------------------------------------------
    # Show saved metric
    # -----------------------------------------------------

    if problem_type == "Classification":

        st.write(
            "**Validation Accuracy:**",
            f"{model_info.get('accuracy', 0) * 100:.2f}%"
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Validation MAE",
            f"{model_info.get('mae', 0):,.2f}"
        )

        col2.metric(
            "Validation RMSE",
            f"{model_info.get('rmse', 0):,.2f}"
        )

        col3.metric(
            "Validation R²",
            f"{model_info.get('r2', 0):.4f}"
        )

    # -----------------------------------------------------
    # Upload test data
    # -----------------------------------------------------

    st.subheader("1️⃣ Upload Test Data")

    test_file = st.file_uploader(
        "Upload CSV containing new market/test data",
        type=["csv"],
        key="test_file"
    )

    if test_file is not None:

        try:
            test_df = pd.read_csv(
                test_file
            )
        except Exception as e:
            st.error(
                f"Could not read test CSV: {e}"
            )
            st.stop()

        st.success(
            f"Test data loaded: "
            f"{test_df.shape[0]} rows"
        )

        st.dataframe(
            test_df.head(20),
            width="stretch"
        )

        # -------------------------------------------------
        # Check required columns
        # -------------------------------------------------

        required_features = model_info[
            "features"
        ]

        missing_features = [
            column
            for column in required_features
            if column not in test_df.columns
        ]

        if missing_features:

            st.error(
                "The test file is missing these columns: "
                + ", ".join(missing_features)
            )

            st.stop()

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        st.subheader("2️⃣ Prediction")

        X_new = test_df[
            required_features
        ].copy()

        try:

            with st.spinner(
                "Generating predictions..."
            ):

                predictions = model.predict(
                    X_new
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()

        result_df = test_df.copy()

        result_df["Prediction"] = predictions

        # -------------------------------------------------
        # Classification probability
        # -------------------------------------------------

        if (
            problem_type == "Classification"
            and hasattr(model, "predict_proba")
        ):

            try:

                probabilities = model.predict_proba(
                    X_new
                )

                max_probability = probabilities.max(
                    axis=1
                )

                result_df["Confidence"] = (
                    max_probability * 100
                ).round(2)

            except Exception:
                pass

        # -------------------------------------------------
        # Show predictions
        # -------------------------------------------------

        st.subheader("🎯 Predictions")

        st.dataframe(
            result_df,
            width="stretch"
        )

        # -------------------------------------------------
        # Download predictions
        # -------------------------------------------------

        csv_data = result_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Predictions",
            data=csv_data,
            file_name="trading_predictions.csv",
            mime="text/csv",
            width="stretch"
        )

        # =================================================
        # CLASSIFICATION TEST PERFORMANCE
        # =================================================

        target_column = model_info[
            "target_column"
        ]

        if (
            problem_type == "Classification"
            and target_column in test_df.columns
        ):

            y_actual = test_df[
                target_column
            ]

            try:

                test_accuracy = accuracy_score(
                    y_actual,
                    predictions
                )

                st.subheader(
                    "📊 Test Data Performance"
                )

                st.metric(
                    "Test Accuracy",
                    f"{test_accuracy * 100:.2f}%"
                )

                report = classification_report(
                    y_actual,
                    predictions,
                    zero_division=0
                )

                st.text(report)

            except Exception as e:

                st.warning(
                    f"Could not calculate classification metrics: {e}"
                )

        # =================================================
        # REGRESSION TEST PERFORMANCE
        # =================================================

        elif (
            problem_type == "Regression"
            and target_column in test_df.columns
        ):

            y_actual = pd.to_numeric(
                test_df[target_column],
                errors="coerce"
            )

            valid = y_actual.notna()

            if valid.sum() > 0:

                actual = y_actual[valid]
                predicted = np.asarray(predictions)[valid]

                test_mae = mean_absolute_error(
                    actual,
                    predicted
                )

                test_mse = mean_squared_error(
                    actual,
                    predicted
                )

                test_rmse = np.sqrt(
                    test_mse
                )

                test_r2 = r2_score(
                    actual,
                    predicted
                )

                st.subheader(
                    "📊 Test Data Performance"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "MAE",
                    f"{test_mae:,.2f}"
                )

                col2.metric(
                    "RMSE",
                    f"{test_rmse:,.2f}"
                )

                col3.metric(
                    "R²",
                    f"{test_r2:.4f}"
                )

                col4.metric(
                    "R² Percentage",
                    f"{test_r2 * 100:.2f}%"
                )

            else:

                st.warning(
                    "The target column does not contain "
                    "valid numeric values for evaluation."
                )

        else:

            st.info(
                "ℹ️ Your test file does not contain the target "
                "column, so actual test performance cannot be "
                "calculated. Predictions are still available."
            )