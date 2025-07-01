# Semantic Similarity Analysis App

A Streamlit application for analyzing semantic similarity between blog posts using embeddings to detect content cannibalization and editorial outliers.

## 🎯 Purpose

This tool helps content managers and SEO specialists:
- Identify potential content cannibalization (similar content competing for the same keywords)
- Detect outlier content that doesn't align with the editorial line
- Optimize content strategy by understanding semantic relationships between pages

## 🚀 Quick Start

### Option 1: Run Locally

1. Clone this repository or download the files
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```
4. Open your browser to `http://localhost:8501`

### Option 2: Deploy on Streamlit Cloud

1. Push the code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy the app by selecting your repository

## 📋 Requirements

Create a `requirements.txt` file with:
```
streamlit==1.28.0
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
openpyxl==3.1.2
```

## 📁 File Structure

```
semantic-similarity-app/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 📊 Input File Format

The app expects an Excel file (.xlsx or .xls) with:
- **Column 1**: URLs of the web pages
- **Column 2**: Embeddings as comma-separated numbers

Example:
| URL | Embeddings |
|-----|------------|
| https://example.com/page1 | 0.123,-0.456,0.789,... |
| https://example.com/page2 | 0.321,-0.654,0.987,... |

## 🔧 How It Works

1. **Upload**: Drag and drop your Excel file
2. **Processing**: The app converts embeddings and calculates cosine similarity
3. **Analysis**: 
   - Shows similarity scores for all URL pairs
   - Identifies potential cannibalization (>85% similarity)
   - Detects outlier content
4. **Download**: Export complete results as CSV

## 📈 Features

### Interactive Analysis
- Adjustable similarity threshold for preview
- Summary statistics
- Top similar pairs visualization

### Automatic Detection
- **Cannibalization Alert**: Pairs with >85% similarity
- **Outlier Detection**: Content that differs significantly from editorial line
- **Statistical Summary**: Average, max, and min similarity scores

### Export Options
- Full CSV export with all URL pairs
- Similarity scores in both decimal and percentage format
- Timestamped filenames for tracking

## 📝 Output CSV Format

The downloaded CSV contains:
- `URL_1`: First URL in the pair
- `URL_2`: Second URL in the pair
- `Similarity_Score`: Raw cosine similarity (0-1)
- `Similarity_Percentage`: Similarity as percentage (0-100%)

## 💡 Interpreting Results

### Similarity Ranges
- **>85%**: High risk of cannibalization - consider consolidating or differentiating
- **70-85%**: Moderate similarity - ensure different keyword targets
- **<40%**: Low similarity - verify content aligns with site theme

### Action Items
1. Review high-similarity pairs for consolidation opportunities
2. Check outliers to ensure brand consistency
3. Use similarity data to inform content planning

## 🐛 Troubleshooting

### Memory Issues
If processing large files (>1000 URLs):
- Consider using Google Colab for more resources
- Process in batches
- Upgrade Streamlit Cloud plan

### File Upload Errors
- Ensure Excel file has exactly 2 columns
- Check embeddings are comma-separated numbers
- Verify no empty rows

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the app!

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) for the web interface
- [scikit-learn](https://scikit-learn.org/) for similarity calculations
- [pandas](https://pandas.pydata.org/) for data manipulation
