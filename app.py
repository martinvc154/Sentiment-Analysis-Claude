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

        # Show mapping summary
        st.divider()
        st.subheader("Mapping Summary")
        mapping_df = pd.DataFrame({
            'Field': ['Organization', 'Question', 'Sentiment'],
            'Mapped Column': [
                st.session_state.org_column or '❌ Not mapped',
                st.session_state.question_column or '❌ Not mapped',
                st.session_state.sentiment_column or '❌ Not mapped'
            ]
        })
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

        # Placeholder messages for other visualizations
        elif chart_type == "Divergence Plot":
            st.info("📊 Divergence Plot visualization coming soon!")
            st.markdown("This will show sentiment divergence across different dimensions.")

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
