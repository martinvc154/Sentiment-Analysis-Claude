import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'text_column': None,
        'id_column': None,
        'org_column': None,
        'question_column': None
    }

# Sidebar
with st.sidebar:
    st.title("📊 Sentiment Analysis")
    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file containing text data for sentiment analysis"
    )

    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df

            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📝 {len(df)} rows, {len(df.columns)} columns")

            # Column mapping section
            st.markdown("---")
            st.subheader("Column Mapping")

            column_options = [''] + list(df.columns)

            # Response Text Column (required)
            text_col = st.selectbox(
                "Response Text Column *",
                options=column_options,
                key='text_column_selector',
                help="Required: Select the column containing text for analysis"
            )
            st.session_state.column_mapping['text_column'] = text_col if text_col else None

            # Interview ID Column (optional)
            id_col = st.selectbox(
                "Interview ID Column",
                options=column_options,
                key='id_column_selector',
                help="Optional: Select the column containing interview IDs"
            )
            st.session_state.column_mapping['id_column'] = id_col if id_col else None

            # Organization Column (optional)
            org_col = st.selectbox(
                "Organization Column",
                options=column_options,
                key='org_column_selector',
                help="Optional: Select the column containing organization names"
            )
            st.session_state.column_mapping['org_column'] = org_col if org_col else None

            # Question Column (optional)
            question_col = st.selectbox(
                "Question Column",
                options=column_options,
                key='question_column_selector',
                help="Optional: Select the column containing questions"
            )
            st.session_state.column_mapping['question_column'] = question_col if question_col else None

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.session_state.uploaded_data = None
    else:
        st.session_state.uploaded_data = None
        st.session_state.column_mapping = {
            'text_column': None,
            'id_column': None,
            'org_column': None,
            'question_column': None
        }

# Main area
if st.session_state.uploaded_data is None:
    # Show instructions when no file is uploaded
    st.title("Welcome to Sentiment Analysis")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### Getting Started

        To begin analyzing sentiment in your data:

        1. 📤 **Upload a CSV file** using the sidebar
        2. 🗂️ **Map your columns** to the appropriate fields
        3. ▶️ **Run the analysis** to get insights

        ---

        #### Required Column
        - **Response Text Column**: The column containing text to analyze

        #### Optional Columns
        - **Interview ID Column**: Unique identifier for each response
        - **Organization Column**: Organization or group information
        - **Question Column**: The question being answered

        ---

        💡 **Tip**: Make sure your CSV file has headers in the first row
        """)
else:
    # Show data preview when file is uploaded
    st.title("Data Preview")

    # Show first 5 rows
    st.dataframe(
        st.session_state.uploaded_data.head(5),
        use_container_width=True
    )

    # Column mapping summary
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Column Mapping Summary")
        mapping = st.session_state.column_mapping

        mapping_data = {
            'Field': ['Response Text Column', 'Interview ID Column', 'Organization Column', 'Question Column'],
            'Mapped To': [
                mapping['text_column'] or '—',
                mapping['id_column'] or '—',
                mapping['org_column'] or '—',
                mapping['question_column'] or '—'
            ],
            'Status': [
                '✅ Required' if mapping['text_column'] else '⚠️ Required',
                '✓ Optional' if mapping['id_column'] else '— Optional',
                '✓ Optional' if mapping['org_column'] else '— Optional',
                '✓ Optional' if mapping['question_column'] else '— Optional'
            ]
        }

        st.table(pd.DataFrame(mapping_data))

    with col2:
        st.subheader("Actions")
        # Run Analysis button (disabled if required column not mapped)
        can_run = st.session_state.column_mapping['text_column'] is not None

        if st.button(
            "▶️ Run Analysis",
            disabled=not can_run,
            use_container_width=True,
            type="primary"
        ):
            st.info("Analysis functionality will be implemented next!")

        if not can_run:
            st.warning("⚠️ Please map the Response Text Column to enable analysis")
