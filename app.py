import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Semantic Similarity Analysis",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Semantic Similarity Analysis for Content Cannibalization")
st.markdown("""
This app analyzes semantic similarity between blog posts using their embeddings to:
- Identify potential content cannibalization (similar content that might compete)
- Detect outlier content that doesn't align with the editorial line
""")

# File upload
uploaded_file = st.file_uploader(
    "Upload your Excel file with URLs and embeddings",
    type=['xlsx', 'xls'],
    help="File should contain URLs in first column and embeddings in second column"
)

if uploaded_file is not None:
    # Load data
    with st.spinner('Loading file...'):
        df = pd.read_excel(uploaded_file)
        
    # Get column names
    url_column = df.columns[0]
    embedding_column = df.columns[1]
    
    st.success(f"✅ File loaded successfully! Found {len(df)} URLs")
    
    # Display basic info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total URLs", len(df))
    with col2:
        st.metric("Embedding Dimensions", len(df[embedding_column].iloc[0].split(',')))
    
    # Process embeddings
    with st.spinner('Processing embeddings and calculating similarities...'):
        # Convert string embeddings to numpy arrays
        embeddings = []
        valid_indices = []
        
        for idx, embedding_str in enumerate(df[embedding_column]):
            try:
                # Convert string to numpy array
                embedding = np.array([float(x) for x in embedding_str.split(',')])
                embeddings.append(embedding)
                valid_indices.append(idx)
            except:
                st.warning(f"⚠️ Could not process embedding for row {idx}")
        
        embeddings = np.array(embeddings)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create results dataframe
        results = []
        n = len(valid_indices)
        
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
                url1_idx = valid_indices[i]
                url2_idx = valid_indices[j]
                
                results.append({
                    'URL_1': df[url_column].iloc[url1_idx],
                    'URL_2': df[url_column].iloc[url2_idx],
                    'Similarity_Score': similarity_matrix[i, j],
                    'Similarity_Percentage': similarity_matrix[i, j] * 100
                })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Sort by similarity score (descending)
        results_df = results_df.sort_values('Similarity_Score', ascending=False)
        
    st.success(f"✅ Analysis complete! Calculated {len(results_df):,} similarity pairs")
    
    # Display options
    st.markdown("### 📊 Analysis Results")
    
    # Filter options
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        min_similarity = st.slider(
            "Minimum Similarity % (for preview)",
            min_value=0,
            max_value=100,
            value=70,
            help="This filter is only for preview. The CSV will contain all pairs."
        )
    
    with col2:
        top_n = st.number_input(
            "Show top N results",
            min_value=10,
            max_value=len(results_df),
            value=min(50, len(results_df)),
            step=10
        )
    
    # Filter for preview
    preview_df = results_df[results_df['Similarity_Percentage'] >= min_similarity].head(int(top_n))
    
    # Display preview
    st.markdown(f"### Preview (showing {len(preview_df)} pairs with similarity ≥ {min_similarity}%)")
    
    # Format the preview dataframe
    preview_display = preview_df.copy()
    preview_display['Similarity_Percentage'] = preview_display['Similarity_Percentage'].round(2).astype(str) + '%'
    preview_display = preview_display[['URL_1', 'URL_2', 'Similarity_Percentage']]
    
    st.dataframe(
        preview_display,
        use_container_width=True,
        height=400
    )
    
    # Summary statistics
    st.markdown("### 📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Similarity", f"{results_df['Similarity_Percentage'].mean():.2f}%")
    with col2:
        st.metric("Max Similarity", f"{results_df['Similarity_Percentage'].max():.2f}%")
    with col3:
        st.metric("Min Similarity", f"{results_df['Similarity_Percentage'].min():.2f}%")
    with col4:
        high_similarity = len(results_df[results_df['Similarity_Percentage'] >= 80])
        st.metric("Pairs > 80% Similar", high_similarity)
    
    # Potential issues
    st.markdown("### 🚨 Potential Issues Detected")
    
    # Cannibalization candidates (>85% similar)
    cannibalization = results_df[results_df['Similarity_Percentage'] >= 85]
    if len(cannibalization) > 0:
        st.warning(f"**Potential Cannibalization:** Found {len(cannibalization)} pairs with >85% similarity")
        with st.expander("View potential cannibalization pairs"):
            cann_display = cannibalization.head(20).copy()
            cann_display['Similarity_Percentage'] = cann_display['Similarity_Percentage'].round(2).astype(str) + '%'
            st.dataframe(cann_display[['URL_1', 'URL_2', 'Similarity_Percentage']])
    else:
        st.success("No severe cannibalization issues detected (no pairs >85% similar)")
    
    # Outliers (calculate average similarity per URL)
    url_avg_similarity = {}
    for url in df[url_column]:
        url_similarities = results_df[
            (results_df['URL_1'] == url) | (results_df['URL_2'] == url)
        ]['Similarity_Score'].values
        if len(url_similarities) > 0:
            url_avg_similarity[url] = np.mean(url_similarities)
    
    # Find outliers (URLs with low average similarity)
    if url_avg_similarity:
        avg_sim_df = pd.DataFrame(list(url_avg_similarity.items()), 
                                  columns=['URL', 'Average_Similarity'])
        avg_sim_df = avg_sim_df.sort_values('Average_Similarity')
        
        # Calculate outlier threshold (e.g., below 1st quartile - 1.5*IQR)
        Q1 = avg_sim_df['Average_Similarity'].quantile(0.25)
        Q3 = avg_sim_df['Average_Similarity'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q1 - 1.5 * IQR
        
        outliers = avg_sim_df[avg_sim_df['Average_Similarity'] < outlier_threshold]
        
        if len(outliers) > 0:
            st.warning(f"**Potential Outliers:** Found {len(outliers)} URLs that may not align with editorial line")
            with st.expander("View potential outlier URLs"):
                outliers_display = outliers.copy()
                outliers_display['Average_Similarity'] = (outliers_display['Average_Similarity'] * 100).round(2).astype(str) + '%'
                st.dataframe(outliers_display)
        else:
            # Show bottom 5 anyway
            st.info("**Least Similar Content:** URLs with lowest average similarity to others")
            with st.expander("View least similar URLs"):
                bottom_5 = avg_sim_df.head(5).copy()
                bottom_5['Average_Similarity'] = (bottom_5['Average_Similarity'] * 100).round(2).astype(str) + '%'
                st.dataframe(bottom_5)
    
    # Download section
    st.markdown("### 💾 Download Results")
    
    # Prepare full results for download
    download_df = results_df.copy()
    download_df['Similarity_Percentage'] = download_df['Similarity_Percentage'].round(4)
    download_df = download_df[['URL_1', 'URL_2', 'Similarity_Score', 'Similarity_Percentage']]
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"similarity_analysis_{timestamp}.csv"
    
    # Download button
    st.download_button(
        label="📥 Download Complete Similarity Analysis (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        help="Download all URL pairs with their similarity scores"
    )
    
    # Additional insights
    st.markdown("### 💡 Next Steps")
    st.info("""
    **How to use the results:**
    1. **High Similarity (>85%)**: Review these pairs for potential content consolidation or differentiation
    2. **Medium Similarity (70-85%)**: Check if these pages target different keywords/intent
    3. **Low Average Similarity**: Review outlier content to ensure it aligns with your editorial strategy
    4. **Use filters in Excel/Google Sheets** to analyze specific similarity ranges
    """)

else:
    # Instructions when no file is uploaded
    st.info("""
    👆 Please upload your Excel file to begin the analysis.
    
    **File requirements:**
    - Excel format (.xlsx or .xls)
    - First column: URLs
    - Second column: Embeddings (comma-separated numbers)
    """)
    
    # Example of expected format
    with st.expander("View expected file format"):
        st.markdown("""
        | URL | Embeddings |
        |-----|------------|
        | https://example.com/page1 | 0.123,-0.456,0.789,... |
        | https://example.com/page2 | 0.321,-0.654,0.987,... |
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for content optimization")
