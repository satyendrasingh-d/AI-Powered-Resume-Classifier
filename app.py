"""
AI-Powered Resume Classifier Streamlit Application
A comprehensive tool to classify resumes into different job categories
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time
from resume_classifier_utils import ResumeClassifier, ResumePreprocessor

# Page configuration
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_classifier(model_type):
    """Load the trained classifier"""
    classifier = ResumeClassifier(model_type=model_type)
    if classifier.load_model('models'):
        return classifier
    return None


def display_prediction_result(result):
    """Display prediction results in a nice format"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.success(f"### 🎯 Predicted Category: **{result['category']}**")
        

    if result['probabilities']:
        # Sort probabilities
        probs_df = pd.DataFrame(
            list(result['probabilities'].items()),
            columns=['Category', 'Confidence']
        ).sort_values('Confidence', ascending=False)

        # Top 5 predictions
        top_5 = probs_df.head(5)

        with col2:
            st.metric(
                "Top Confidence Score",
                f"{top_5.iloc[0]['Confidence']:.2%}"
            )

        # Visualize top 5
        st.subheader("📊 Top 5 Predictions")
        fig = px.bar(
            top_5,
            x='Confidence',
            y='Category',
            orientation='h',
            color='Confidence',
            color_continuous_scale='Blues',
            labels={'Confidence': 'Confidence Score', 'Category': 'Job Category'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Full probability table
        with st.expander("📋 View All Predictions"):
            st.dataframe(
                probs_df.assign(Confidence=probs_df['Confidence'].apply(lambda x: f"{x:.2%}")),
                use_container_width=True,
                hide_index=True
            )


def main():
    # Header
    st.title("📄 AI-Powered Resume Classifier")
    st.markdown(
        "**Classify resumes into job categories using machine learning models**"
    )

    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")

    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Classification Model",
        options=['logistic', 'naive_bayes', 'knn'],
        format_func=lambda x: {
            'logistic': '🔵 Logistic Regression (Recommended)',
            'naive_bayes': '🟡 Multinomial Naive Bayes',
            'knn': '🟢 K-Nearest Neighbors'
        }[x],
        help="Choose the ML model for classification"
    )

    # Load classifier
    classifier = load_classifier(model_option)

    if classifier is None:
        st.error(
            "⚠️ Models not found. Please run `train_model.py` first to train the models."
        )
        st.info(
            "**Setup Instructions:**\n"
            "1. Run: `python train_model.py /path/to/your/dataset`\n"
            "2. This will train all models and save them\n"
            "3. Then run: `streamlit run app.py`"
        )
        return

    # Display model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Information")
    categories = classifier.get_categories()
    st.sidebar.metric("Total Categories", len(categories))
    st.sidebar.metric("Model Type", model_option.replace('_', ' ').title())

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Classify Resume", "📚 Category Guide", "ℹ️ About"])

    # Tab 1: Classification
    with tab1:
        st.subheader("Upload or Paste Your Resume")

        col1, col2 = st.columns([1, 1])

        resume_text = None
        input_method = st.radio(
            "Choose input method:",
            options=["Paste Text", "Upload File"],
            horizontal=True
        )

        if input_method == "Paste Text":
            resume_text = st.text_area(
                "Paste your resume content here:",
                height=250,
                placeholder="Enter resume text...",
                label_visibility="collapsed"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a resume file (.txt, .pdf, or .docx)",
                type=['txt', 'pdf', 'docx'],
                label_visibility="collapsed"
            )
            if uploaded_file:
                try:
                    if uploaded_file.type == 'text/plain':
                        resume_text = uploaded_file.read().decode('utf-8')
                    elif uploaded_file.type == 'application/pdf':
                        try:
                            import PyPDF2
                            pdf_reader = PyPDF2.PdfReader(uploaded_file)
                            resume_text = "\n".join(
                                page.extract_text()
                                for page in pdf_reader.pages
                            )
                        except ImportError:
                            st.warning("PDF support requires PyPDF2. Install with: pip install PyPDF2")
                    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                        try:
                            from docx import Document
                            doc = Document(uploaded_file)
                            resume_text = "\n".join(
                                para.text for para in doc.paragraphs
                            )
                        except ImportError:
                            st.warning("DOCX support requires python-docx. Install with: pip install python-docx")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        # Classification button
        if resume_text:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("🚀 Classify Resume", use_container_width=True):
                    with st.spinner("Analyzing resume..."):
                        # Show preprocessing info
                        preprocessor = ResumePreprocessor()
                        processed = preprocessor.preprocess(resume_text)

                        with st.expander("📝 Preprocessing Details"):
                            col_a, col_b = st.columns([1, 1])
                            with col_a:
                                st.write("**Original text length:** " + str(len(resume_text)))
                            with col_b:
                                st.write("**Processed text length:** " + str(len(processed)))

                            words = processed.split()
                            st.write("**Unique words:** " + str(len(set(words))))
                            st.write("**Total words:** " + str(len(words)))

                            st.write("**Sample processed text:**")
                            st.code(' '.join(words[:50]) + "...")

                        time.sleep(0.5)  # Brief delay for UX

                        # Get prediction
                        result = classifier.predict(resume_text)

                        st.markdown("---")
                        display_prediction_result(result)

        else:
            st.info("👆 Enter or upload a resume to get started")

    # Tab 2: Category Guide
    with tab2:
        st.subheader("Available Job Categories")
        st.write(f"Total of **{len(categories)}** job categories supported:")

        # Display categories in columns
        cols = st.columns(3)
        for idx, category in enumerate(sorted(categories)):
            with cols[idx % 3]:
                st.write(f"• {category}")

    # Tab 3: About
    with tab3:
        st.subheader("About This Application")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            ### 🎯 Purpose
            This application uses machine learning to automatically classify resumes
            into different job categories. It helps HR teams and recruiters quickly
            categorize incoming resumes.

            ### 🛠️ Technology Stack
            - **ML Framework:** Scikit-learn
            - **Text Processing:** NLTK
            - **Web Framework:** Streamlit
            - **Visualization:** Plotly
            """)

        with col2:
            st.markdown("""
            ### 📊 Model Performance
            The classification models achieve:
            - **Accuracy:** ~98-99%
            - **Models:** Logistic Regression, Naive Bayes, KNN
            - **Features:** TF-IDF Vectorization
            - **Training Data:** 962 resumes

            ### ⚙️ Preprocessing Steps
            1. Remove URLs
            2. Clean special characters
            3. Convert to lowercase
            4. Remove stopwords
            5. Tokenization
            6. Stemming
            """)

        st.markdown("---")
        st.markdown("""
        ### 📖 How to Use
        1. Select a classification model from the sidebar
        2. Either paste resume text or upload a file
        3. Click "Classify Resume" to get predictions
        4. View confidence scores for all categories
        5. Use the Category Guide to explore available job categories
        """)


if __name__ == "__main__":
    main()
