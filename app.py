import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'org_column' not in st.session_state:
    st.session_state.org_column = None
if 'question_column' not in st.session_state:
    st.session_state.question_column = None
if 'sentiment_column' not in st.session_state:
    st.session_state.sentiment_column = None
if 'response_column' not in st.session_state:
    st.session_state.response_column = None
if 'confidence_column' not in st.session_state:
    st.session_state.confidence_column = None
if 'emotion_columns' not in st.session_state:
    st.session_state.emotion_columns = {
        'joy': None, 'anger': None, 'fear': None,
        'sadness': None, 'surprise': None, 'disgust': None
    }

# Create exports directory if it doesn't exist
if not os.path.exists('exports'):
    os.makedirs('exports')

# Title
st.title("📊 Sentiment Analysis Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "🔗 Column Mapping", "📈 Visualizations"])

# TAB 1: Data Upload
with tab1:
    st.header("Upload Your Data")
    st.markdown("Upload a CSV or Excel file containing your sentiment analysis data.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with your sentiment data"
    )

    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.data = df
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")

            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Show basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    elif st.session_state.data is not None:
        st.info("✅ Data already loaded. Upload a new file to replace it.")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)

# TAB 2: Column Mapping
with tab2:
    st.header("Map Your Columns")
    st.markdown("Select which columns in your dataset correspond to organizations, questions, and sentiment scores.")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the 'Data Upload' tab first.")
    else:
        df = st.session_state.data
        columns = df.columns.tolist()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Organization Column")
            org_col = st.selectbox(
                "Select the column containing organization names:",
                options=['None'] + columns,
                index=0 if st.session_state.org_column is None else columns.index(st.session_state.org_column) + 1 if st.session_state.org_column in columns else 0,
                key='org_select'
            )
            if org_col != 'None':
                st.session_state.org_column = org_col
                st.info(f"Unique organizations: {df[org_col].nunique()}")
            else:
                st.session_state.org_column = None

        with col2:
            st.subheader("Question Column")
            q_col = st.selectbox(
                "Select the column containing questions:",
                options=['None'] + columns,
                index=0 if st.session_state.question_column is None else columns.index(st.session_state.question_column) + 1 if st.session_state.question_column in columns else 0,
                key='question_select'
            )
            if q_col != 'None':
                st.session_state.question_column = q_col
                st.info(f"Unique questions: {df[q_col].nunique()}")
            else:
                st.session_state.question_column = None

        with col3:
            st.subheader("Sentiment Column")
            sent_col = st.selectbox(
                "Select the column containing sentiment scores:",
                options=['None'] + columns,
                index=0 if st.session_state.sentiment_column is None else columns.index(st.session_state.sentiment_column) + 1 if st.session_state.sentiment_column in columns else 0,
                key='sentiment_select'
            )
            if sent_col != 'None':
                st.session_state.sentiment_column = sent_col
                # Try to show statistics if numeric
                try:
                    st.info(f"Range: [{df[sent_col].min():.2f}, {df[sent_col].max():.2f}]")
                except:
                    st.info("Non-numeric values detected")
            else:
                st.session_state.sentiment_column = None

        # Additional columns for Divergence Plot
        st.divider()
        st.markdown("### Additional Columns (Optional - for Divergence Plot)")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Response Text Column")
            response_col = st.selectbox(
                "Select the column containing response text:",
                options=['None'] + columns,
                index=0 if st.session_state.response_column is None else columns.index(st.session_state.response_column) + 1 if st.session_state.response_column in columns else 0,
                key='response_select',
                help="Text responses to display in hover tooltips"
            )
            if response_col != 'None':
                st.session_state.response_column = response_col
            else:
                st.session_state.response_column = None

        with col2:
            st.subheader("Confidence Column")
            conf_col = st.selectbox(
                "Select the column containing confidence scores:",
                options=['None'] + columns,
                index=0 if st.session_state.confidence_column is None else columns.index(st.session_state.confidence_column) + 1 if st.session_state.confidence_column in columns else 0,
                key='confidence_select',
                help="Confidence scores (0-1) to scale point sizes"
            )
            if conf_col != 'None':
                st.session_state.confidence_column = conf_col
                try:
                    st.info(f"Range: [{df[conf_col].min():.2f}, {df[conf_col].max():.2f}]")
                except:
                    st.info("Non-numeric values detected")
            else:
                st.session_state.confidence_column = None

        # Emotion columns mapping
        st.subheader("Emotion Columns")
        st.markdown("Map columns for each emotion type (all required for Divergence Plot):")

        emotion_cols = st.columns(3)
        emotion_names = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust']

        for idx, emotion in enumerate(emotion_names):
            with emotion_cols[idx % 3]:
                emotion_col = st.selectbox(
                    f"{emotion.capitalize()}:",
                    options=['None'] + columns,
                    index=0 if st.session_state.emotion_columns[emotion] is None else columns.index(st.session_state.emotion_columns[emotion]) + 1 if st.session_state.emotion_columns[emotion] in columns else 0,
                    key=f'{emotion}_select'
                )
                if emotion_col != 'None':
                    st.session_state.emotion_columns[emotion] = emotion_col
                else:
                    st.session_state.emotion_columns[emotion] = None

        # Show mapping summary
        st.divider()
        st.subheader("Mapping Summary")

        # Basic mappings
        mapping_data = {
            'Field': ['Organization', 'Question', 'Sentiment', 'Response Text', 'Confidence'],
            'Mapped Column': [
                st.session_state.org_column or '❌ Not mapped',
                st.session_state.question_column or '❌ Not mapped',
                st.session_state.sentiment_column or '❌ Not mapped',
                st.session_state.response_column or '⚠️ Optional (for Divergence Plot)',
                st.session_state.confidence_column or '⚠️ Optional (for Divergence Plot)'
            ]
        }

        # Add emotion mappings
        for emotion in emotion_names:
            mapping_data['Field'].append(f'{emotion.capitalize()} Emotion')
            mapping_data['Mapped Column'].append(
                st.session_state.emotion_columns[emotion] or '⚠️ Optional (for Divergence Plot)'
            )

        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

# TAB 3: Visualizations
with tab3:
    st.header("Visualizations")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the 'Data Upload' tab first.")
    else:
        # Chart selector
        st.subheader("Select Visualization Type")
        chart_type = st.radio(
            "Choose a visualization:",
            options=["Heatmap", "Divergence Plot", "Emotion Profiles", "Clusters"],
            horizontal=True,
            help="Select the type of visualization you want to create"
        )

        st.divider()

        # HEATMAP VISUALIZATION
        if chart_type == "Heatmap":
            # Check if required columns are mapped
            if st.session_state.org_column is None or st.session_state.question_column is None:
                st.error("❌ Both Organization and Question columns must be mapped to display the heatmap.")
                st.info("👈 Please go to the 'Column Mapping' tab and map both columns.")
            elif st.session_state.sentiment_column is None:
                st.error("❌ Sentiment column must be mapped to display the heatmap.")
                st.info("👈 Please go to the 'Column Mapping' tab and map the sentiment column.")
            else:
                df = st.session_state.data
                org_col = st.session_state.org_column
                q_col = st.session_state.question_column
                sent_col = st.session_state.sentiment_column

                try:
                    # Create pivot table with average sentiment scores
                    pivot_data = df.pivot_table(
                        values=sent_col,
                        index=org_col,
                        columns=q_col,
                        aggfunc='mean'
                    )

                    if pivot_data.empty:
                        st.error("❌ No data available for heatmap. Please check your data and column mappings.")
                    else:
                        # Replace NaN with 0 for clustering
                        pivot_filled = pivot_data.fillna(0)

                        # Perform hierarchical clustering
                        if len(pivot_filled) > 1 and len(pivot_filled.columns) > 1:
                            # Cluster rows (organizations)
                            row_linkage = hierarchy.linkage(
                                pdist(pivot_filled, metric='euclidean'),
                                method='ward'
                            )
                            row_dendro = hierarchy.dendrogram(row_linkage, no_plot=True)
                            row_order = row_dendro['leaves']

                            # Cluster columns (questions)
                            col_linkage = hierarchy.linkage(
                                pdist(pivot_filled.T, metric='euclidean'),
                                method='ward'
                            )
                            col_dendro = hierarchy.dendrogram(col_linkage, no_plot=True)
                            col_order = col_dendro['leaves']

                            # Reorder the data based on clustering
                            pivot_clustered = pivot_filled.iloc[row_order, col_order]
                        else:
                            pivot_clustered = pivot_filled
                            row_dendro = None
                            col_dendro = None

                        # Create the heatmap using Plotly
                        st.subheader("Sentiment Heatmap with Hierarchical Clustering")

                        # Create dendrogram figures if clustering was performed
                        if row_dendro is not None and col_dendro is not None:
                            # Create figure with dendrograms
                            fig = ff.create_dendrogram(
                                pivot_filled.values,
                                orientation='left',
                                labels=pivot_filled.index.tolist()
                            )
                            fig.update_layout(width=200, height=600)

                            # For simplicity, show heatmap without integrated dendrograms
                            # (Plotly's create_dendrogram doesn't easily integrate with heatmaps)
                            # Instead, create a standalone heatmap with clustered data

                            fig = go.Figure(data=go.Heatmap(
                                z=pivot_clustered.values,
                                x=pivot_clustered.columns.tolist(),
                                y=pivot_clustered.index.tolist(),
                                colorscale='RdYlGn',
                                zmid=0,
                                zmin=-1,
                                zmax=1,
                                colorbar=dict(title="Sentiment Score"),
                                text=np.round(pivot_clustered.values, 2),
                                texttemplate='%{text}',
                                textfont={"size": 10},
                                hovertemplate='Organization: %{y}<br>Question: %{x}<br>Sentiment: %{z:.2f}<extra></extra>'
                            ))

                            fig.update_layout(
                                title="Sentiment Heatmap (Hierarchically Clustered)",
                                xaxis_title="Questions",
                                yaxis_title="Organizations",
                                height=max(400, len(pivot_clustered) * 30),
                                width=max(800, len(pivot_clustered.columns) * 50),
                                xaxis={'side': 'bottom'},
                                yaxis={'side': 'left'}
                            )
                        else:
                            # Simple heatmap without clustering (not enough data points)
                            fig = go.Figure(data=go.Heatmap(
                                z=pivot_clustered.values,
                                x=pivot_clustered.columns.tolist(),
                                y=pivot_clustered.index.tolist(),
                                colorscale='RdYlGn',
                                zmid=0,
                                zmin=-1,
                                zmax=1,
                                colorbar=dict(title="Sentiment Score"),
                                text=np.round(pivot_clustered.values, 2),
                                texttemplate='%{text}',
                                textfont={"size": 10},
                                hovertemplate='Organization: %{y}<br>Question: %{x}<br>Sentiment: %{z:.2f}<extra></extra>'
                            ))

                            fig.update_layout(
                                title="Sentiment Heatmap",
                                xaxis_title="Questions",
                                yaxis_title="Organizations",
                                height=max(400, len(pivot_clustered) * 30),
                                width=max(800, len(pivot_clustered.columns) * 50)
                            )

                        st.plotly_chart(fig, use_container_width=True)

                        # Store the pivot data for export
                        st.session_state.heatmap_data = pivot_clustered
                        st.session_state.row_linkage = row_linkage if len(pivot_filled) > 1 else None
                        st.session_state.col_linkage = col_linkage if len(pivot_filled.columns) > 1 else None

                        # EXPORT SECTION
                        st.divider()
                        st.subheader("Export Options")

                        col1, col2, col3 = st.columns([2, 2, 2])

                        with col1:
                            dpi = st.selectbox(
                                "Select DPI:",
                                options=[150, 300, 600],
                                index=1,
                                help="Higher DPI = better quality but larger file size"
                            )

                        with col2:
                            size_options = {
                                "7x5 inches": (7, 5),
                                "10x7 inches": (10, 7),
                                "14x10 inches": (14, 10)
                            }
                            size_label = st.selectbox(
                                "Select Size:",
                                options=list(size_options.keys()),
                                index=1
                            )
                            fig_size = size_options[size_label]

                        with col3:
                            st.write("")  # Spacing
                            st.write("")  # Spacing

                        # Export buttons
                        col1, col2 = st.columns(2)

                        def create_publication_heatmap(data, row_linkage, col_linkage, figsize, dpi):
                            """Create a publication-quality heatmap with dendrograms using matplotlib"""

                            # Set font to Times New Roman
                            matplotlib.rcParams['font.family'] = 'serif'
                            matplotlib.rcParams['font.serif'] = ['Times New Roman']
                            matplotlib.rcParams['font.size'] = 12
                            matplotlib.rcParams['axes.linewidth'] = 2

                            # Create figure with subplots for dendrograms and heatmap
                            if row_linkage is not None and col_linkage is not None:
                                fig = plt.figure(figsize=figsize, dpi=dpi)

                                # Create grid spec
                                gs = fig.add_gridspec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1],
                                                     hspace=0.05, wspace=0.05,
                                                     left=0.15, right=0.95, top=0.95, bottom=0.15)

                                # Top dendrogram (columns)
                                ax_top = fig.add_subplot(gs[0, 1])
                                dendro_top = hierarchy.dendrogram(
                                    col_linkage,
                                    ax=ax_top,
                                    color_threshold=0,
                                    above_threshold_color='black',
                                    no_labels=True
                                )
                                ax_top.set_xticks([])
                                ax_top.set_yticks([])
                                ax_top.spines['top'].set_visible(False)
                                ax_top.spines['right'].set_visible(False)
                                ax_top.spines['bottom'].set_visible(False)
                                ax_top.spines['left'].set_visible(False)

                                # Left dendrogram (rows)
                                ax_left = fig.add_subplot(gs[1, 0])
                                dendro_left = hierarchy.dendrogram(
                                    row_linkage,
                                    ax=ax_left,
                                    orientation='left',
                                    color_threshold=0,
                                    above_threshold_color='black',
                                    no_labels=True
                                )
                                ax_left.set_xticks([])
                                ax_left.set_yticks([])
                                ax_left.spines['top'].set_visible(False)
                                ax_left.spines['right'].set_visible(False)
                                ax_left.spines['bottom'].set_visible(False)
                                ax_left.spines['left'].set_visible(False)

                                # Main heatmap
                                ax_heatmap = fig.add_subplot(gs[1, 1])
                            else:
                                # Simple heatmap without dendrograms
                                fig, ax_heatmap = plt.subplots(figsize=figsize, dpi=dpi)

                            # Create heatmap
                            im = ax_heatmap.imshow(
                                data.values,
                                cmap='RdYlGn',
                                aspect='auto',
                                vmin=-1,
                                vmax=1,
                                interpolation='nearest'
                            )

                            # Set ticks and labels
                            ax_heatmap.set_xticks(np.arange(len(data.columns)))
                            ax_heatmap.set_yticks(np.arange(len(data.index)))
                            ax_heatmap.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=10)
                            ax_heatmap.set_yticklabels(data.index, fontsize=10)

                            # Add gridlines
                            ax_heatmap.set_xticks(np.arange(len(data.columns)) - 0.5, minor=True)
                            ax_heatmap.set_yticks(np.arange(len(data.index)) - 0.5, minor=True)
                            ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=2)

                            # Add colorbar
                            cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
                            cbar.set_label('Sentiment Score', rotation=270, labelpad=25, fontsize=12, weight='bold')
                            cbar.outline.set_linewidth(2)

                            # Add cell values
                            for i in range(len(data.index)):
                                for j in range(len(data.columns)):
                                    value = data.values[i, j]
                                    if not np.isnan(value):
                                        text_color = 'white' if abs(value) > 0.5 else 'black'
                                        ax_heatmap.text(j, i, f'{value:.2f}',
                                                       ha='center', va='center',
                                                       color=text_color, fontsize=8, weight='bold')

                            # Labels
                            ax_heatmap.set_xlabel('Questions', fontsize=14, weight='bold')
                            ax_heatmap.set_ylabel('Organizations', fontsize=14, weight='bold')

                            # Title
                            if row_linkage is not None and col_linkage is not None:
                                fig.suptitle('Sentiment Heatmap with Hierarchical Clustering',
                                           fontsize=16, weight='bold', y=0.98)
                            else:
                                ax_heatmap.set_title('Sentiment Heatmap', fontsize=16, weight='bold', pad=20)

                            # Adjust spines
                            for spine in ax_heatmap.spines.values():
                                spine.set_linewidth(2)

                            plt.tight_layout()
                            return fig

                        with col1:
                            if st.button("📄 Export as PDF", use_container_width=True):
                                with st.spinner("Generating PDF..."):
                                    try:
                                        fig = create_publication_heatmap(
                                            st.session_state.heatmap_data,
                                            st.session_state.row_linkage,
                                            st.session_state.col_linkage,
                                            fig_size,
                                            dpi
                                        )

                                        # Save to file
                                        output_path = 'exports/sentiment_heatmap.pdf'
                                        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=dpi)
                                        plt.close(fig)

                                        # Provide download button
                                        with open(output_path, 'rb') as f:
                                            st.download_button(
                                                label="⬇️ Download PDF",
                                                data=f,
                                                file_name="sentiment_heatmap.pdf",
                                                mime="application/pdf",
                                                use_container_width=True
                                            )
                                        st.success(f"✅ PDF saved to {output_path}")
                                    except Exception as e:
                                        st.error(f"❌ Error creating PDF: {str(e)}")

                        with col2:
                            if st.button("🖼️ Export as PNG", use_container_width=True):
                                with st.spinner("Generating PNG..."):
                                    try:
                                        fig = create_publication_heatmap(
                                            st.session_state.heatmap_data,
                                            st.session_state.row_linkage,
                                            st.session_state.col_linkage,
                                            fig_size,
                                            dpi
                                        )

                                        # Save to file
                                        output_path = 'exports/sentiment_heatmap.png'
                                        fig.savefig(output_path, format='png', bbox_inches='tight', dpi=dpi)
                                        plt.close(fig)

                                        # Provide download button
                                        with open(output_path, 'rb') as f:
                                            st.download_button(
                                                label="⬇️ Download PNG",
                                                data=f,
                                                file_name="sentiment_heatmap.png",
                                                mime="image/png",
                                                use_container_width=True
                                            )
                                        st.success(f"✅ PNG saved to {output_path}")
                                    except Exception as e:
                                        st.error(f"❌ Error creating PNG: {str(e)}")

                except Exception as e:
                    st.error(f"❌ Error creating heatmap: {str(e)}")
                    st.info("Please check that your sentiment column contains numeric values.")

        # DIVERGENCE PLOT VISUALIZATION
        elif chart_type == "Divergence Plot":
            # Check if required columns are mapped
            emotions_mapped = all(st.session_state.emotion_columns[e] is not None
                                 for e in ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust'])

            if st.session_state.sentiment_column is None:
                st.error("❌ Sentiment column must be mapped to display the Divergence Plot.")
                st.info("👈 Please go to the 'Column Mapping' tab and map the sentiment column.")
            elif not emotions_mapped:
                st.error("❌ All emotion columns (Joy, Anger, Fear, Sadness, Surprise, Disgust) must be mapped to display the Divergence Plot.")
                st.info("👈 Please go to the 'Column Mapping' tab and map all emotion columns.")
            else:
                df = st.session_state.data.copy()
                sent_col = st.session_state.sentiment_column

                # Get emotion columns
                emotion_cols = st.session_state.emotion_columns

                # Optional columns
                response_col = st.session_state.response_column
                conf_col = st.session_state.confidence_column
                org_col = st.session_state.org_column

                try:
                    # Calculate emotion valence for each row
                    # emotion_valence = (joy + surprise - anger - fear - sadness - disgust) / 6 (normalized to -1 to 1)
                    df['emotion_valence'] = (
                        df[emotion_cols['joy']] +
                        df[emotion_cols['surprise']] -
                        df[emotion_cols['anger']] -
                        df[emotion_cols['fear']] -
                        df[emotion_cols['sadness']] -
                        df[emotion_cols['disgust']]
                    )

                    # Normalize to -1 to 1 range (assuming emotion scores are 0-1)
                    # Max possible: (1+1-0-0-0-0) = 2, Min possible: (0+0-1-1-1-1) = -4
                    # To normalize to -1 to 1, we'll use: (valence + 4) / 6 * 2 - 1
                    # Simpler: valence / 3 (ranges from -4/3 to 2/3, roughly -1.33 to 0.67)
                    # Better normalization: map [-4, 2] to [-1, 1]
                    df['emotion_valence'] = (df['emotion_valence'] + 4) / 6 * 2 - 1

                    # Get dominant emotion for each row
                    emotion_names = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust']
                    df['dominant_emotion'] = df[list(emotion_cols.values())].idxmax(axis=1)
                    # Map column names back to emotion names
                    reverse_emotion_map = {v: k for k, v in emotion_cols.items()}
                    df['dominant_emotion'] = df['dominant_emotion'].map(reverse_emotion_map)

                    # Create color palette for emotions
                    emotion_colors = {
                        'joy': '#FFD700',      # Gold
                        'surprise': '#FF69B4',  # Hot Pink
                        'anger': '#FF4500',     # Orange Red
                        'fear': '#8B008B',      # Dark Magenta
                        'sadness': '#4169E1',   # Royal Blue
                        'disgust': '#228B22'    # Forest Green
                    }

                    # Determine color scheme (by organization or by dominant emotion)
                    if org_col is not None:
                        # Use organization for coloring
                        color_by = org_col
                        use_emotion_colors = False
                    else:
                        # Use dominant emotion for coloring
                        color_by = 'dominant_emotion'
                        use_emotion_colors = True

                    # Create scatter plot with marginal histograms
                    from plotly.subplots import make_subplots

                    st.subheader("Sentiment-Emotion Divergence Scatter Plot")

                    # Prepare hover text
                    hover_text = []
                    for idx, row in df.iterrows():
                        # Response preview (first 100 chars)
                        if response_col:
                            response_preview = str(row[response_col])[:100]
                            if len(str(row[response_col])) > 100:
                                response_preview += "..."
                        else:
                            response_preview = "N/A"

                        org_text = f"<br>Organization: {row[org_col]}" if org_col else ""

                        # Format emotions
                        emotions_text = "<br>Emotions: " + ", ".join([
                            f"{e.capitalize()}: {row[emotion_cols[e]]:.2f}"
                            for e in emotion_names
                        ])

                        hover_text.append(
                            f"<b>Response:</b> {response_preview}{org_text}"
                            f"<br><b>Sentiment:</b> {row[sent_col]:.2f}"
                            f"<br><b>Emotion Valence:</b> {row['emotion_valence']:.2f}"
                            f"{emotions_text}"
                            f"<br><b>Dominant Emotion:</b> {row['dominant_emotion'].capitalize()}"
                        )

                    df['hover_text'] = hover_text

                    # Point sizes based on confidence
                    if conf_col:
                        # Scale confidence to point sizes (e.g., 5 to 30)
                        sizes = df[conf_col] * 25 + 5
                    else:
                        sizes = [15] * len(df)  # Default size

                    # Create figure with marginal histograms
                    fig = make_subplots(
                        rows=2, cols=2,
                        column_widths=[0.85, 0.15],
                        row_heights=[0.15, 0.85],
                        horizontal_spacing=0.01,
                        vertical_spacing=0.01,
                        specs=[
                            [{"type": "histogram"}, None],
                            [{"type": "scatter"}, {"type": "histogram"}]
                        ]
                    )

                    # Main scatter plot (bottom left)
                    if use_emotion_colors:
                        # Color by dominant emotion
                        for emotion in emotion_names:
                            mask = df['dominant_emotion'] == emotion
                            if mask.any():
                                fig.add_trace(
                                    go.Scatter(
                                        x=df[mask][sent_col],
                                        y=df[mask]['emotion_valence'],
                                        mode='markers',
                                        name=emotion.capitalize(),
                                        marker=dict(
                                            size=sizes[mask],
                                            color=emotion_colors[emotion],
                                            line=dict(color='white', width=0.5),
                                            opacity=0.7
                                        ),
                                        hovertext=df[mask]['hover_text'],
                                        hoverinfo='text',
                                        showlegend=True
                                    ),
                                    row=2, col=1
                                )
                    else:
                        # Color by organization
                        for org in df[org_col].unique():
                            mask = df[org_col] == org
                            fig.add_trace(
                                go.Scatter(
                                    x=df[mask][sent_col],
                                    y=df[mask]['emotion_valence'],
                                    mode='markers',
                                    name=str(org),
                                    marker=dict(
                                        size=sizes[mask],
                                        line=dict(color='white', width=0.5),
                                        opacity=0.7
                                    ),
                                    hovertext=df[mask]['hover_text'],
                                    hoverinfo='text',
                                    showlegend=True
                                ),
                                row=2, col=1
                            )

                    # Add reference lines (dashed gray)
                    # Vertical line at x=0
                    fig.add_shape(
                        type="line",
                        x0=0, y0=-1, x1=0, y1=1,
                        line=dict(color="gray", width=2, dash="dash"),
                        row=2, col=1
                    )

                    # Horizontal line at y=0
                    fig.add_shape(
                        type="line",
                        x0=-1, y0=0, x1=1, y1=0,
                        line=dict(color="gray", width=2, dash="dash"),
                        row=2, col=1
                    )

                    # Highlight bottom-right quadrant (Suppressed Negativity) with light red background
                    fig.add_shape(
                        type="rect",
                        x0=0, y0=-1, x1=1, y1=0,
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        line=dict(width=0),
                        layer="below",
                        row=2, col=1
                    )

                    # Add quadrant labels as annotations
                    quadrant_labels = [
                        dict(x=0.5, y=0.5, text="<b>Authentic Positive</b><br>(+,+)", showarrow=False,
                             font=dict(size=12, color="green"), bgcolor="rgba(255,255,255,0.7)"),
                        dict(x=-0.5, y=0.5, text="<b>Strategic Optimism</b><br>(-,+)", showarrow=False,
                             font=dict(size=12, color="orange"), bgcolor="rgba(255,255,255,0.7)"),
                        dict(x=-0.5, y=-0.5, text="<b>Authentic Negative</b><br>(-,-)", showarrow=False,
                             font=dict(size=12, color="red"), bgcolor="rgba(255,255,255,0.7)"),
                        dict(x=0.5, y=-0.5, text="<b>Suppressed Negativity</b><br>(+,-)", showarrow=False,
                             font=dict(size=12, color="darkred"), bgcolor="rgba(255,200,200,0.7)")
                    ]

                    for label in quadrant_labels:
                        fig.add_annotation(
                            label,
                            row=2, col=1,
                            xref="x", yref="y"
                        )

                    # Top marginal histogram (sentiment distribution)
                    fig.add_trace(
                        go.Histogram(
                            x=df[sent_col],
                            nbinsx=30,
                            marker=dict(color='steelblue', opacity=0.6),
                            showlegend=False,
                            hovertemplate='Sentiment: %{x}<br>Count: %{y}<extra></extra>'
                        ),
                        row=1, col=1
                    )

                    # Right marginal histogram (emotion valence distribution)
                    fig.add_trace(
                        go.Histogram(
                            y=df['emotion_valence'],
                            nbinsy=30,
                            marker=dict(color='coral', opacity=0.6),
                            showlegend=False,
                            hovertemplate='Emotion Valence: %{y}<br>Count: %{x}<extra></extra>'
                        ),
                        row=2, col=2
                    )

                    # Update axes
                    fig.update_xaxes(title_text="Sentiment Score", range=[-1, 1], row=2, col=1)
                    fig.update_yaxes(title_text="Emotion Valence", range=[-1, 1], row=2, col=1)
                    fig.update_xaxes(showticklabels=False, range=[-1, 1], row=1, col=1)
                    fig.update_yaxes(showticklabels=False, range=[-1, 1], row=2, col=2)

                    # Update layout
                    fig.update_layout(
                        title="Sentiment-Emotion Divergence Analysis",
                        height=700,
                        width=900,
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.05
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )

                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)

                    # Store data for export
                    st.session_state.divergence_data = df
                    st.session_state.divergence_fig = fig

                    # EXPORT SECTION
                    st.divider()
                    st.subheader("Export Options")

                    col1, col2, col3 = st.columns([2, 2, 2])

                    with col1:
                        dpi = st.selectbox(
                            "Select DPI:",
                            options=[150, 300, 600],
                            index=1,
                            help="Higher DPI = better quality but larger file size",
                            key='divergence_dpi'
                        )

                    with col2:
                        size_options = {
                            "7x5 inches": (7, 5),
                            "10x7 inches": (10, 7),
                            "14x10 inches": (14, 10)
                        }
                        size_label = st.selectbox(
                            "Select Size:",
                            options=list(size_options.keys()),
                            index=1,
                            key='divergence_size'
                        )
                        fig_size = size_options[size_label]

                    with col3:
                        st.write("")  # Spacing
                        st.write("")  # Spacing

                    # Export buttons
                    col1, col2 = st.columns(2)

                    def create_publication_divergence_plot(df, sent_col, org_col, conf_col,
                                                          emotion_cols, use_emotion_colors,
                                                          figsize, dpi):
                        """Create a publication-quality divergence plot using matplotlib"""

                        # Set font to Times New Roman
                        matplotlib.rcParams['font.family'] = 'serif'
                        matplotlib.rcParams['font.serif'] = ['Times New Roman']
                        matplotlib.rcParams['font.size'] = 10
                        matplotlib.rcParams['axes.linewidth'] = 1.5

                        # Create figure with gridspec for marginal histograms
                        fig = plt.figure(figsize=figsize, dpi=dpi)
                        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                             hspace=0.05, wspace=0.05,
                                             left=0.1, right=0.9, top=0.92, bottom=0.1)

                        # Main scatter plot
                        ax_main = fig.add_subplot(gs[1, 0])

                        # Top histogram
                        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)

                        # Right histogram
                        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

                        # Point sizes
                        if conf_col:
                            sizes = df[conf_col] * 200 + 20
                        else:
                            sizes = [50] * len(df)

                        # Emotion colors
                        emotion_colors_map = {
                            'joy': '#FFD700',
                            'surprise': '#FF69B4',
                            'anger': '#FF4500',
                            'fear': '#8B008B',
                            'sadness': '#4169E1',
                            'disgust': '#228B22'
                        }

                        # Plot scatter points
                        if use_emotion_colors:
                            emotion_names = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust']
                            for emotion in emotion_names:
                                mask = df['dominant_emotion'] == emotion
                                if mask.any():
                                    ax_main.scatter(
                                        df[mask][sent_col],
                                        df[mask]['emotion_valence'],
                                        s=sizes[mask] if conf_col else 50,
                                        c=emotion_colors_map[emotion],
                                        label=emotion.capitalize(),
                                        alpha=0.6,
                                        edgecolors='white',
                                        linewidth=0.5
                                    )
                        else:
                            scatter = ax_main.scatter(
                                df[sent_col],
                                df['emotion_valence'],
                                s=sizes if conf_col else 50,
                                c=df[org_col].astype('category').cat.codes if org_col else 'steelblue',
                                cmap='tab10',
                                alpha=0.6,
                                edgecolors='white',
                                linewidth=0.5
                            )
                            if org_col:
                                # Add legend for organizations
                                handles = [plt.scatter([], [], s=50, c=scatter.cmap(scatter.norm(i)),
                                          label=org, alpha=0.6, edgecolors='white', linewidth=0.5)
                                         for i, org in enumerate(df[org_col].unique())]
                                ax_main.legend(handles=handles, loc='upper left', fontsize=8)

                        # Highlight bottom-right quadrant
                        ax_main.add_patch(plt.Rectangle((0, -1), 1, 1,
                                         facecolor='red', alpha=0.1, zorder=0))

                        # Add reference lines
                        ax_main.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax_main.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

                        # Add quadrant labels
                        ax_main.text(0.5, 0.5, 'Authentic Positive\n(+,+)',
                                   ha='center', va='center', fontsize=9, weight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                   color='green')
                        ax_main.text(-0.5, 0.5, 'Strategic Optimism\n(-,+)',
                                   ha='center', va='center', fontsize=9, weight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                   color='orange')
                        ax_main.text(-0.5, -0.5, 'Authentic Negative\n(-,-)',
                                   ha='center', va='center', fontsize=9, weight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                   color='red')
                        ax_main.text(0.5, -0.5, 'Suppressed Negativity\n(+,-)',
                                   ha='center', va='center', fontsize=9, weight='bold',
                                   bbox=dict(boxstyle='round', facecolor='rgba(255,200,200,0.9)'),
                                   color='darkred')

                        # Set limits and labels
                        ax_main.set_xlim(-1, 1)
                        ax_main.set_ylim(-1, 1)
                        ax_main.set_xlabel('Sentiment Score', fontsize=12, weight='bold')
                        ax_main.set_ylabel('Emotion Valence', fontsize=12, weight='bold')
                        ax_main.grid(True, alpha=0.3, linestyle=':')

                        if use_emotion_colors:
                            ax_main.legend(loc='upper left', fontsize=8, framealpha=0.9)

                        # Top histogram (sentiment distribution)
                        ax_top.hist(df[sent_col], bins=30, color='steelblue', alpha=0.6, edgecolor='black')
                        ax_top.set_xlim(-1, 1)
                        ax_top.set_ylabel('Count', fontsize=9)
                        ax_top.tick_params(axis='x', labelbottom=False)
                        ax_top.grid(True, alpha=0.3, linestyle=':')

                        # Right histogram (emotion valence distribution)
                        ax_right.hist(df['emotion_valence'], bins=30, orientation='horizontal',
                                     color='coral', alpha=0.6, edgecolor='black')
                        ax_right.set_ylim(-1, 1)
                        ax_right.set_xlabel('Count', fontsize=9)
                        ax_right.tick_params(axis='y', labelleft=False)
                        ax_right.grid(True, alpha=0.3, linestyle=':')

                        # Title
                        fig.suptitle('Sentiment-Emotion Divergence Analysis',
                                   fontsize=14, weight='bold', y=0.98)

                        return fig

                    with col1:
                        if st.button("📄 Export as PDF", use_container_width=True, key='divergence_pdf'):
                            with st.spinner("Generating PDF..."):
                                try:
                                    fig_export = create_publication_divergence_plot(
                                        st.session_state.divergence_data,
                                        sent_col,
                                        org_col,
                                        conf_col,
                                        emotion_cols,
                                        use_emotion_colors,
                                        fig_size,
                                        dpi
                                    )

                                    # Save to file
                                    output_path = 'exports/divergence_plot.pdf'
                                    fig_export.savefig(output_path, format='pdf', bbox_inches='tight', dpi=dpi)
                                    plt.close(fig_export)

                                    # Provide download button
                                    with open(output_path, 'rb') as f:
                                        st.download_button(
                                            label="⬇️ Download PDF",
                                            data=f,
                                            file_name="divergence_plot.pdf",
                                            mime="application/pdf",
                                            use_container_width=True,
                                            key='divergence_pdf_download'
                                        )
                                    st.success(f"✅ PDF saved to {output_path}")
                                except Exception as e:
                                    st.error(f"❌ Error creating PDF: {str(e)}")

                    with col2:
                        if st.button("🖼️ Export as PNG", use_container_width=True, key='divergence_png'):
                            with st.spinner("Generating PNG..."):
                                try:
                                    fig_export = create_publication_divergence_plot(
                                        st.session_state.divergence_data,
                                        sent_col,
                                        org_col,
                                        conf_col,
                                        emotion_cols,
                                        use_emotion_colors,
                                        fig_size,
                                        dpi
                                    )

                                    # Save to file
                                    output_path = 'exports/divergence_plot.png'
                                    fig_export.savefig(output_path, format='png', bbox_inches='tight', dpi=dpi)
                                    plt.close(fig_export)

                                    # Provide download button
                                    with open(output_path, 'rb') as f:
                                        st.download_button(
                                            label="⬇️ Download PNG",
                                            data=f,
                                            file_name="divergence_plot.png",
                                            mime="image/png",
                                            use_container_width=True,
                                            key='divergence_png_download'
                                        )
                                    st.success(f"✅ PNG saved to {output_path}")
                                except Exception as e:
                                    st.error(f"❌ Error creating PNG: {str(e)}")

                except Exception as e:
                    st.error(f"❌ Error creating divergence plot: {str(e)}")
                    st.info("Please check that all emotion and sentiment columns contain numeric values in the 0-1 range.")

        elif chart_type == "Emotion Profiles":
            st.info("😊 Emotion Profiles visualization coming soon!")
            st.markdown("This will display emotion distributions and profiles.")

        elif chart_type == "Clusters":
            st.info("🔍 Clusters visualization coming soon!")
            st.markdown("This will show clustering analysis of sentiment patterns.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Sentiment Analysis Dashboard | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
