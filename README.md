# Movie Recommendation System

A content-based movie recommendation system built with Python and Streamlit that suggests similar movies based on their features using cosine similarity.

## Features

- Content-based filtering using movie metadata (genres, overview, cast, etc.)
- Interactive web interface powered by Streamlit
- Movie poster fetching from TMDB API
- Cosine similarity algorithm for movie recommendations
- Dataset of 10,000 top movies from TMDB

## Prerequisites

- Python 3.7+
- TMDB API Key (get one from [TMDB](https://www.themoviedb.org/settings/api))

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Movie recommendation system"
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
```

3. Install required packages:
```bash
pip install pandas scikit-learn streamlit requests python-dotenv
```

4. Create a `.env` file in the root directory and add your TMDB API key:
```
TMDB_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit app:
```bash
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`

## Project Structure

```
Movie recommendation system/
├── main.py                    # Main application file
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore file
├── .env                       # Environment variables (not tracked)
├── data/
│   └── raw/
│       └── top10K-TMDB-movies.csv  # Movie dataset
└── notebooks/
    └── exploartion.ipynb      # Data exploration notebook
```

## How It Works

1. **Data Loading**: Loads movie dataset from CSV file
2. **Feature Engineering**: Combines relevant features (genres, keywords, cast, crew, overview)
3. **Vectorization**: Converts text features into numerical vectors using CountVectorizer
4. **Similarity Calculation**: Computes cosine similarity between movie vectors
5. **Recommendation**: Returns top N most similar movies based on similarity scores

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning (CountVectorizer, cosine similarity)
- **Streamlit**: Web application framework
- **TMDB API**: Fetching movie posters and additional data
- **python-dotenv**: Environment variable management

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to fork this project and submit pull requests with improvements!