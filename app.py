import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.pipeline import Pipeline  # Keep this for type checking if needed
from pycaret.classification import load_model as pycaret_load_model  # Alias to avoid name conflict
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Adult Income Dataset Dashboard", layout="wide")

# Custom CSS for overall layout, font sizes, buttons, and messages
st.markdown("""
<style>
    /* General spacing for main content */
    .st-emotion-cache-z5fcl4 { /* Target the main content area */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5%; /* Adjust left/right padding for wide layout */
        padding-right: 5%; /* Adjust left/right padding for wide layout */
    }

    /* Increase font size for headers and general text */
    h1, h2, h3, h4, h5, h6 {
        font-size: calc(1.325rem + 0.9vw); /* Responsive font size for headers */
    }
    p, li, .st-emotion-cache-1fcpjtk, .st-emotion-cache-j90zef { /* Targeting common text elements */
        font-size: 1.1rem; /* General text font size */
    }

    /* Spacing between Streamlit elements */
    .stVerticalBlock > div:not(:last-child) {
        margin-bottom: 1.5rem; /* Space between vertical blocks like text, charts, tables */
    }

    /* Specific for sidebar radio buttons */
    .stRadio > label {
        font-size: 18px; /* Already set, keeping it */
    }

    /* Adjust spacing for columns if needed */
    .st-emotion-cache-1c7y2kl {
        gap: 2rem; /* Gap between columns */
    }

    /* Customizing all Streamlit buttons */
    div.stButton > button {
        background-color: #706D54; /* User's requested color */
        color: black; /* Text color for buttons */
        border: none;
        padding: 0.75rem 1.25rem; /* Standard button padding */
        border-radius: 0.5rem; /* Slightly rounded corners */
        transition: all 0.2s ease-in-out; /* Smooth transition for hover effects */
    }
    div.stButton > button:hover {
        background-color: #5a5745; /* Slightly darker shade on hover */
        color: white; /* Adjust text color on hover for better contrast */
    }

    /* Customizing Streamlit messages (info, success, warning, error) */
    /* This targets the main container of Streamlit's status widgets */
    [data-testid="stStatusWidget"] {
        background-color: #706D54; /* Base background for all messages */
        color: black; /* Text color for messages */
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #706D54; /* A subtle border */
    }

    /* Specific styling for different message types to maintain distinction */
    [data-testid="stStatusWidget-Success"] {
        border-left: 5px solid #4CAF50; /* A green border for success */
        color: black; /* Explicitly set text color for success messages */
    }
    [data-testid="stStatusWidget-Info"] {
        border-left: 5px solid #2196F3; /* A blue border for info */
    }
    [data-testid="stStatusWidget-Warning"] {
        background-color: #706D54; /* Apply theme background explicitly to warning messages */
        border-left: 5px solid #ff9800; /* An orange border for warning */
    }
    [data-testid="stStatusWidget-Error"] {
        border-left: 5px solid #f44336; /* A red border for error */
    }

</style>
""", unsafe_allow_html=True)

# Load model summary CSV
@st.cache_data
def load_summary():
    df = pd.read_csv("summary.csv")
    return df.sort_values(by="Accuracy", ascending=False)

# Prediction function, now expects a PyCaret pipeline
def predict(pipeline, input_df):
    if not isinstance(pipeline, Pipeline):
        st.error(f"Error: Loaded object is not a PyCaret Pipeline. Type: {type(pipeline)}")
        st.stop()
    if isinstance(input_df, pd.DataFrame):
        # The PyCaret pipeline will handle preprocessing and prediction
        return pipeline.predict(input_df)
    else:
        raise ValueError("Input must be a pandas DataFrame")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Ranking", "Prediction", "Model Evaluation"])

# --- HOME PAGE ---
if page == "Home":
    st.title("Welcome to the Adult Income Dataset AutoML Dashboard!")
    st.write("""
    This interactive dashboard allows you to explore, predict, and evaluate machine learning models for the UCI Adult Income dataset.

    **Features:**
    - **Model Ranking:** View the performance of various pre-trained models on the Adult Income dataset.
    - **Prediction:** Get real-time income predictions (<=50K or >50K) based on individual or batch data input.
    - **Model Evaluation:** Explore detailed performance metrics and visualizations for all trained models.

    Use the sidebar to navigate between different functionalities.
    """)

    st.header("Key Insights from Model Performance")
    summary_df = load_summary()

    if not summary_df.empty:
        total_models = len(summary_df)
        best_model_name = summary_df.iloc[0]['Model']
        best_model_accuracy = summary_df.iloc[0]['Accuracy']

        # --- Top 3 Models and their Accuracies ---
        top3 = summary_df.head(3)
        # Abbreviate model names for display
        def abbreviate_model(name):
            abbr_map = {
                'Random Forest Classifier': 'RF',
                'Extreme Gradient Boosting': 'XGB',
                'Light Gradient Boosting': 'LGBM',
                'Gradient Boosting': 'GB',
                'AdaBoost': 'ADA',
                'Ridge Classifier': 'Ridge',
                'Logistic Regression': 'LR',
                'Decision Tree': 'DT',
                'K Neighbors': 'KNN',
                'Naive Bayes': 'NB',
                'SVM': 'SVM',
                'MLP': 'MLP'
            }
            for k, v in abbr_map.items():
                if k.lower() in name.lower():
                    return v
            # If not found, return first 6 chars
            return name[:6]

        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown(f"""
                <div style='border:2px solid #636EFA; border-radius:12px; padding:1em; text-align:center; background:#f6f8fa;'>
                    <div style='font-weight:bold; font-size:1.1em;'>1st Model</div>
                    <div style='font-size:2em; color:#636EFA;'>{abbreviate_model(str(top3.iloc[0]['Model']))}</div>
                    <div style='font-size:1.1em;'>Accuracy: <b>{top3.iloc[0]['Accuracy']:.2%}</b></div>
                </div>
            """, unsafe_allow_html=True)
        with colB:
            if len(top3) > 1:
                st.markdown(f"""
                    <div style='border:2px solid #00CC96; border-radius:12px; padding:1em; text-align:center; background:#f6f8fa;'>
                        <div style='font-weight:bold; font-size:1.1em;'>2nd Model</div>
                        <div style='font-size:2em; color:#00CC96;'>{abbreviate_model(str(top3.iloc[1]['Model']))}</div>
                        <div style='font-size:1.1em;'>Accuracy: <b>{top3.iloc[1]['Accuracy']:.2%}</b></div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='border:2px solid #00CC96; border-radius:12px; padding:1em; text-align:center; background:#f6f8fa;'>-</div>", unsafe_allow_html=True)
        with colC:
            if len(top3) > 2:
                st.markdown(f"""
                    <div style='border:2px solid #AB63FA; border-radius:12px; padding:1em; text-align:center; background:#f6f8fa;'>
                        <div style='font-weight:bold; font-size:1.1em;'>3rd Model</div>
                        <div style='font-size:2em; color:#AB63FA;'>{abbreviate_model(str(top3.iloc[2]['Model']))}</div>
                        <div style='font-size:1.1em;'>Accuracy: <b>{top3.iloc[2]['Accuracy']:.2%}</b></div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='border:2px solid #AB63FA; border-radius:12px; padding:1em; text-align:center; background:#f6f8fa;'>-</div>", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Models Trained", value=total_models)
        with col2:
            st.metric(label="Best Model Accuracy", value=f"{best_model_accuracy:.2%}", help=f"The best performing model is {best_model_name}")

        # New KPIs for accuracy thresholds
        models_above_85 = summary_df[summary_df['Accuracy'] >= 0.85]
        count_above_85 = len(models_above_85)

        models_above_80 = summary_df[summary_df['Accuracy'] >= 0.80]
        count_above_80 = len(models_above_80)
        
        st.write("\n") # Add some space
        col3, col4 = st.columns(2)
        with col3:
            st.metric(label="Models with Accuracy >= 85%", value=count_above_85)
        with col4:
            st.metric(label="Models with Accuracy >= 80%", value=count_above_80)

        st.subheader("Top 5 Models by Accuracy")
        for index, row in summary_df.head(5).iterrows():
            st.write(f"**{row['Model']}:** {row['Accuracy']:.2%}")
    else:
        st.info("No model summary data available. Please ensure 'summary.csv' exists and contains data.")

# --- MODEL RANKING PAGE ---
elif page == "Model Ranking":
    st.title("Model Ranking Dashboard")
    df = load_summary()
    st.dataframe(df, use_container_width=True)

# --- PREDICTION PAGE ---
elif page == "Prediction":
    st.title("Model Prediction")

    # Check if 'Model' directory exists (adapted from 'Models')
    model_dir = "Model"
    if not os.path.exists(model_dir):
        st.error(f"The '{model_dir}' folder was not found. Please ensure it exists and contains your .pkl model files.")
        st.stop()

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not model_files:
        st.error(f"No .pkl model files found in the '{model_dir}' folder. Please ensure your trained models are saved there.")
        st.stop()

    model_name_map = {
        "RandomForestClassifier_model.pkl": "Random Forest Classifier",
        "XGBClassifier.pkl": "Extreme Gradient Boosting",
        "LGBMClassifier.pkl": "Light Gradient Boosting",
        "GradientBoostingClassifier.pkl": "Gradient Boosting",
        "AdaBoostClassifier.pkl": "AdaBoost",
        "RidgeClassifier.pkl": "Ridge Classifier"
    }

    # Create a list of display names for the selectbox, with a placeholder
    display_names = ["--- Select a Model ---"] + [model_name_map.get(f, f.replace(".pkl", "")) for f in model_files]

    selected_model_display_name = st.selectbox(
        "Select a trained model",
        options=display_names,
        index=0 # Default to the placeholder
    )

    # If the placeholder is selected, stop and ask the user to select a model
    if selected_model_display_name == "--- Select a Model ---":
        st.warning("Please select a model from the dropdown to proceed with predictions.")
        st.stop()

    # Map the selected display name back to its .pkl filename
    selected_model_file = None
    for filename, display_name in model_name_map.items():
        if display_name == selected_model_display_name:
            selected_model_file = filename
            break
    if selected_model_file is None:
        # Fallback: try to match by removing .pkl
        for f in model_files:
            if f.replace(".pkl", "") == selected_model_display_name:
                selected_model_file = f
                break

    # Fallback if somehow mapping fails (shouldn't happen if model_name_map is complete)
    if selected_model_file is None:
        st.error("Could not find the corresponding model file. Please try again or check your model files.")
        st.stop()
    
    model_pipeline = None
    try:
        # PyCaret expects the model name without .pkl
        model_path = os.path.join(model_dir, selected_model_file.replace(".pkl", ""))
        model_pipeline = pycaret_load_model(model_path)
        st.success("Model loaded successfully!")
        st.write(f"Loaded model pipeline type: {type(model_pipeline)}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Make sure the model file is not corrupted and compatible with your PyCaret version.")
        st.stop()

    input_method = st.radio("Select Input Method", ["Form Input", "Upload CSV"])

    feature_columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education.num',
        'marital.status', 'occupation', 'relationship', 'sex',
        'hours.per.week', 'native.country'
    ]

    if input_method == "Form Input":
        st.subheader("Enter Individual Data")
        age = st.number_input("Age", 0, 120, 30)
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
        fnlwgt = st.number_input("Final Weight (fnlwgt)", 0, 1000000, 100000)
        education = st.selectbox("Education", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
        education_num = st.number_input("Education Num", 1, 16, 9)
        marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
        occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
        relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
        sex = st.selectbox("Sex", ["Female", "Male"])
        hours_per_week = st.number_input("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])

        input_df = pd.DataFrame([{
            'age': age,
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education': education,
            'education.num': education_num,
            'marital.status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'sex': sex,
            'hours.per.week': hours_per_week,
            'native.country': native_country
        }])
        
        input_df = input_df[feature_columns]

        if st.button("Predict"):
            try:
                pred = predict(model_pipeline, input_df)
                if pred[0] == 1:
                    st.success("Your salary is not <=50K")
                else:
                    st.success("Your salary is <=50K")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    else:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            csv_df = pd.read_csv(uploaded_file)
            
            missing_columns = [col for col in feature_columns if col not in csv_df.columns]
            if missing_columns:
                st.error(f"Missing required feature columns in CSV: {', '.join(missing_columns)}. Please ensure your CSV has all the necessary input features.")
                st.stop()
            
            csv_df = csv_df[feature_columns]
            
            st.write("Uploaded Data Preview:")
            st.dataframe(csv_df)

            if st.button("Predict from CSV"):
                try:
                    predictions = predict(model_pipeline, csv_df)
                    csv_df["Prediction"] = ["Your salary is not <=50K" if p==1 else "Your salary is <=50K" for p in predictions]
                    st.write("Prediction Results:")
                    st.dataframe(csv_df)
                    csv_download = csv_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results", csv_download, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")

# --- MODEL EVALUATION PAGE ---
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("Explore detailed performance metrics and visualizations for all trained models.")

    summary_df = load_summary()

    if not summary_df.empty:
        # --- Top 3 Metrics for Top 5 Models ---
        st.subheader("Top 5 Models - Key Metrics")
        top_5_models = summary_df.head(5)
        metrics_to_plot = ['Accuracy', 'AUC', 'F1']

        for metric in metrics_to_plot:
            fig = px.bar(
                top_5_models,
                x='Model',
                y=metric,
                title=f'{metric} for Top 5 Models',
                labels={'Model': 'Model', metric: metric},
                height=400,
                color_discrete_sequence=['#4F6D7A', '#7CA982', '#A786DF'] # Muted blue, green, purple
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='black',
                title_font_color='black',
                legend_title_font_color='black'
            )
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)

        # --- Comparison Visualizations for All Models ---
        st.subheader("All Models - Performance Comparison")

        # Chart 1: Accuracy of All Models
        fig_accuracy_all = px.bar(
            summary_df,
            x='Model',
            y='Accuracy',
            title='Accuracy of All Models',
            labels={'Model': 'Model', 'Accuracy': 'Accuracy'},
            height=500,
            color_discrete_sequence=['#4F6D7A', '#7CA982', '#A786DF']
        )
        fig_accuracy_all.update_layout(
            xaxis_tickangle=-45,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black',
            title_font_color='black',
            legend_title_font_color='black'
        )
        fig_accuracy_all.update_xaxes(tickfont_color="black", title_font_color="black")
        fig_accuracy_all.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig_accuracy_all, use_container_width=True)

        # Chart 2: Accuracy vs AUC for All Models (Scatter Plot)
        fig_scatter_auc = px.scatter(
            summary_df,
            x='Accuracy',
            y='AUC',
            hover_name='Model',
            size='F1', # Size points by F1-score for additional insight
            title='Accuracy vs. AUC for All Models',
            labels={'Accuracy': 'Accuracy', 'AUC': 'AUC'},
            height=500,
            color_discrete_sequence=['#4F6D7A', '#7CA982', '#A786DF']
        )
        fig_scatter_auc.update_layout(
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black',
            title_font_color='black',
            legend_title_font_color='black'
        )
        fig_scatter_auc.update_xaxes(tickfont_color="black", title_font_color="black")
        fig_scatter_auc.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig_scatter_auc, use_container_width=True)

        # Chart 3: Box Plot of Accuracies
        fig_box = px.box(
            summary_df,
            y='Accuracy',
            points='all',
            title='Box Plot of Model Accuracies',
            color_discrete_sequence=['#4F6D7A'],
            height=400
        )
        fig_box.update_layout(
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black',
            title_font_color='black',
            legend_title_font_color='black'
        )
        fig_box.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info("No model summary data available for evaluation. Please ensure 'summary.csv' exists and contains data.") 