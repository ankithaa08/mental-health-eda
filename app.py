import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Mental Health in Tech EDA", layout="wide")

# Hide warnings
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Load and clean data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset.csv')
        
        # Handle Age Outliers
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df = df[(df['Age'] >= 18) & (df['Age'] <= 75)]
        
        # Standardize Gender
        df['Gender'] = df['Gender'].str.lower().str.strip()
        male_terms = ['male', 'm', 'man', 'cis male', 'male-ish', 'maile', 'mal', 'male (cis)', 'make', 'male ', 'msle', 'mail', 'malr']
        female_terms = ['female', 'f', 'woman', 'cis female', 'femake', 'female ', 'cis-female/femme', 'femail']
        
        def clean_gender(gender):
            if pd.isna(gender):
                return 'Other'
            if gender in male_terms:
                return 'Male'
            elif gender in female_terms:
                return 'Female'
            else:
                return 'Other'
                
        df['Gender'] = df['Gender'].apply(clean_gender)
        
        # Handle missing values
        df['self_employed'].fillna('No', inplace=True)
        df['work_interfere'].fillna("Don't know", inplace=True)
        
        # Drop columns
        df.drop(['comments', 'state', 'Timestamp'], axis=1, inplace=True, errors='ignore')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Overview", "Data Overview", "Visualizations", "Correlation & Conclusion"])

    if page == "Project Overview":
        st.title("🧠 Mental Health in Tech - Exploratory Data Analysis")
        st.markdown("""
        ### Project Summary
        This project conducts an Exploratory Data Analysis (EDA) on the Mental Health in Tech Survey dataset from 2014. 
        The dataset contains survey responses from tech workers regarding their mental health and workplace support.
        
        ### Business Objective
        To provide actionable recommendations for tech companies to improve mental health benefits, reduce stigma, and foster a supportive workplace.
        """)

    elif page == "Data Overview":
        st.title("📊 Data Overview & Wrangling")
        st.write("### Raw Data Preview (First 5 Rows)")
        st.dataframe(df.head())
        
        st.write("### Dataset Dimensions")
        st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
        
        st.write("### Data Wrangling Applied")
        st.markdown("""
        - Filtered 'Age' to realistic values (18-75) to remove unrealistic outliers.
        - Standardized 'Gender' into 'Male', 'Female', and 'Other' due to the huge variety of free-text responses.
        - Filled missing 'self_employed' values with 'No' (the mode).
        - Dropped 'comments', 'state', and 'Timestamp' as they were mostly missing or irrelevant to demographic trends.
        """)

    elif page == "Visualizations":
        st.title("📈 Data Visualizations")
        
        viz_choice = st.selectbox("Choose a chart to view:", [
            "1. Gender Distribution", 
            "2. Age Distribution", 
            "3. Seeking Treatment",
            "4. Family History vs Treatment",
            "5. Employer Wellness Programs",
            "6. Remote Work vs Treatment"
        ])
        
        if viz_choice == "1. Gender Distribution":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Gender', data=df, palette='Set2', ax=ax)
            ax.set_title('Gender Distribution')
            st.pyplot(fig)
            st.info("**Insight:** The tech industry sample is predominantly male.")
            
        elif viz_choice == "2. Age Distribution":
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df['Age'], bins=20, kde=True, color='skyblue', ax=ax)
            ax.set_title('Age Distribution')
            st.pyplot(fig)
            st.info("**Insight:** Most respondents are between 25 and 35 years old.")
            
        elif viz_choice == "3. Seeking Treatment":
            fig, ax = plt.subplots(figsize=(6, 6))
            df['treatment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], ax=ax)
            ax.set_title('Proportion of Respondents Seeking Treatment')
            ax.set_ylabel('')
            st.pyplot(fig)
            st.info("**Insight:** Roughly half of the respondents have sought treatment for mental health.")
            
        elif viz_choice == "4. Family History vs Treatment":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='family_history', hue='treatment', data=df, palette='Set1', ax=ax)
            ax.set_title('Treatment Seeking Based on Family History')
            st.pyplot(fig)
            st.info("**Insight:** Individuals with a family history of mental illness are significantly more likely to seek treatment.")
            
        elif viz_choice == "5. Employer Wellness Programs":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='wellness_program', data=df, palette='pastel', ax=ax)
            ax.set_title('Does Employer Offer a Wellness Program?')
            st.pyplot(fig)
            st.info("**Insight:** Most employers do not have a wellness program, or employees don't know about it.")
            
        elif viz_choice == "6. Remote Work vs Treatment":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='remote_work', hue='treatment', data=df, palette='Greens', ax=ax)
            ax.set_title('Remote Work vs Treatment')
            st.pyplot(fig)
            st.info("**Insight:** The ratio of seeking treatment is roughly similar regardless of remote work status.")

    elif page == "Correlation & Conclusion":
        st.title("🔍 Correlation & Conclusion")
        
        st.write("### Correlation Heatmap")
        le = LabelEncoder()
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(exclude=['number']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=False, ax=ax)
        st.pyplot(fig)
        st.info("**Insight:** 'Treatment' has stronger correlations with 'family_history' and 'work_interfere'. There is also a relationship between 'benefits', 'care_options', and 'wellness_program'.")
        
        st.write("### 🚀 Business Recommendations")
        st.markdown("""
        - **Increase Awareness:** A massive number of employees answer 'Don't know' regarding benefits, wellness programs, and leave. HR must actively communicate these policies.
        - **Destigmatize Mental Health:** Data shows extreme fear of negative consequences for discussing mental health. Leadership should openly discuss mental wellness to normalize it.
        - **Train Supervisors:** Employees are hesitant to speak to supervisors. Managerial training on empathy and handling mental health disclosures is critical.
        """)
else:
    st.warning("Please ensure 'dataset.csv' is in the same directory as this app to continue.")
