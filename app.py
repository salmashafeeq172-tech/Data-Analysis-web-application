import os
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df = None
model = None
columns = []


# ----------- HOME PAGE -----------
@app.route("/", methods=["GET", "POST"])
def index():
    global df, columns
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            if file.filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            columns = df.columns.tolist()
            session["columns"] = columns
            return redirect(url_for("eda"))
    return render_template("index.html")


# ----------- EDA PAGE -----------
@app.route("/eda")
def eda():
    global df
    if df is None:
        return redirect(url_for("index"))

    # Summary and missing values tables
    summary = df.describe(include="all").to_html(classes="table table-bordered")
    missing = df.isnull().sum().to_frame("Missing Values").to_html(classes="table table-bordered")

    # Interactive plots
    plots = []

    # Histogram for each numeric column
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        plots.append(fig.to_html(full_html=False))

    # Correlation heatmap
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )
        plots.append(fig_corr.to_html(full_html=False))

    # Optional: scatter matrix
    if len(num_cols) > 1 and len(num_cols) <= 6:  # limit to avoid huge plots
        fig_matrix = px.scatter_matrix(df, dimensions=num_cols, title="Scatter Matrix")
        plots.append(fig_matrix.to_html(full_html=False))

    return render_template("eda.html", tables=[summary, missing], plots=plots)



# ----------- INTERACTIVE PLOT PAGE -----------
@app.route("/plot", methods=["GET", "POST"])
def plot():
    global df
    if df is None:
        return redirect(url_for("index"))

    plot_div = None
    if request.method == "POST":
        x_col = request.form["x_col"]
        y_col = request.form.get("y_col")
        plot_type = request.form["plot_type"]
        color = request.form.get("color", "#007bff")

        if plot_type == "scatter" and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        elif plot_type == "line" and y_col:
            fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        elif plot_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        elif plot_type == "hist":
            fig = px.histogram(df, x=x_col, nbins=30, color_discrete_sequence=[color])
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=[color])

        plot_div = fig.to_html(full_html=False)

    return render_template("plot.html", columns=session["columns"], plot_div=plot_div)



# ----------- MODEL TRAINING -----------
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

@app.route("/model", methods=["GET", "POST"])
def model_train():
    global df, model
    if df is None:
        return redirect(url_for("index"))

    message = None
    plots = []
    if request.method == "POST":
        target = request.form["target"]
        model_type = request.form["model_type"]

        X = df.drop(columns=[target])
        y = df[target]

        # --- Handle missing values ---
        X = X.copy()
        X = X.fillna(X.median(numeric_only=True))
        y = y.fillna(y.median() if y.dtype != 'object' else y.mode()[0])

        # --- Label encode categorical columns ---
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # --- Impute any remaining missing values numerically ---
        imputer = SimpleImputer(strategy="mean")
        X[X.columns] = imputer.fit_transform(X)

        # --- Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "linear":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = model.score(X_test, y_test)

        # Save model, encoders, and imputer
        joblib.dump({"model": model, "encoders": label_encoders, "imputer": imputer, "features": X.columns.tolist()}, "model.pkl")

        session["features"] = X.columns.tolist()
        message = f"{model_type.capitalize()} model trained successfully with R² score: {score:.2f}"

        # --- Residual plot ---
        residuals = y_test - y_pred
        fig_resid = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residuals"}, title="Residuals Plot")
        plots.append(fig_resid.to_html(full_html=False))

        # --- Feature importance ---
        if model_type == "rf":
            importance = model.feature_importances_
            feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
            fig_imp = px.bar(feat_imp, x="Feature", y="Importance", title="Feature Importances", color="Importance", color_continuous_scale="Blues")
            plots.append(fig_imp.to_html(full_html=False))

    return render_template("model.html", columns=session["columns"], message=message, plots=plots)




# ----------- PREDICT -----------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    global df

    prediction = None
    results_table = None
    download_link = None

    # Load model, encoders, and imputer
    try:
        saved_data = joblib.load("model.pkl")
    except FileNotFoundError:
        return redirect(url_for("model_train"))

    model = saved_data["model"]
    label_encoders = saved_data["encoders"]
    imputer = saved_data["imputer"]
    features = saved_data["features"]

    # --- Build feature values for dropdowns ---
    feature_values = {}
    if df is not None:
        for col in features:
            if col in df.columns:
                unique_vals = df[col].dropna().unique().tolist()
                if len(unique_vals) <= 20 and df[col].dtype == 'object':
                    feature_values[col] = unique_vals
                else:
                    feature_values[col] = None
            else:
                feature_values[col] = None
    else:
        # fallback: if df not available, assume numeric
        feature_values = {col: None for col in features}

    if request.method == "POST":
        action = request.form.get("action")

        # --- Manual Input Prediction ---
        if action == "manual":
            input_dict = {}
            for col in features:
                val = request.form.get(col, 0)
                input_dict[col] = val

            X_pred = pd.DataFrame([input_dict])

            # Apply label encoding to categorical columns
            for col, le in label_encoders.items():
                if col in X_pred.columns:
                    val = X_pred.at[0, col]
                    if val in le.classes_:
                        X_pred.at[0, col] = le.transform([val])[0]
                    else:
                        X_pred.at[0, col] = 0

            # Convert numeric fields to float
            for col in X_pred.columns:
                try:
                    X_pred[col] = X_pred[col].astype(float)
                except ValueError:
                    X_pred[col] = 0.0

            # Apply imputer
            X_pred[X_pred.columns] = imputer.transform(X_pred)

            # Predict
            prediction = model.predict(X_pred)[0]

        # --- File Upload Prediction ---
        elif action == "file":
            file = request.files.get("file")
            if file:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)

                test_df = pd.read_csv(filepath) if file.filename.endswith(".csv") else pd.read_excel(filepath)

                for col, le in label_encoders.items():
                    if col in test_df.columns:
                        test_df[col] = test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

                missing_cols = set(features) - set(test_df.columns)
                for col in missing_cols:
                    test_df[col] = 0

                test_df = test_df[features]
                test_df[test_df.columns] = imputer.transform(test_df)
                preds = model.predict(test_df)

                test_df["Prediction"] = preds
                results_table = test_df.to_html(classes="table table-bordered", index=False)
                out_path = os.path.join("uploads", "predictions.csv")
                test_df.to_csv(out_path, index=False)
                download_link = f"/uploads/predictions.csv"

    return render_template(
        "predict.html",
        features=features,
        feature_values=feature_values,
        prediction=prediction,
        results_table=results_table,
        download_link=download_link
    )







if __name__ == "__main__":
    app.run(debug=True)

