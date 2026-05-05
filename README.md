# ⚽ Advanced AI Football Predictor

An end-to-end Machine Learning pipeline that predicts football match outcomes, expected goals, and highlights value bets using XGBoost.

This project encompasses data scraping, feature engineering, model training, an API backend (FastAPI), and an interactive frontend dashboard (Streamlit) to present high-confidence match predictions and betting suggestions.

## 🌟 Features

- **Automated Data Pipeline:** Fetches upcoming fixtures and historical match data.
- **Machine Learning Models:** 
  - **Outcome Classifier:** XGBoost multiclass classifier predicts Home Win, Draw, or Away Win probabilities.
  - **Goals Regressors:** XGBoost regressors with Poisson objectives estimate expected goals for both teams.
- **Betting Engine (`suggest_bets.py`):** Automatically identifies and suggests the best value bets for platforms like SportyBet based on calculated model confidence and projected goal totals.
- **RESTful API (`FastAPI`):** Programmatic access to the predictions, enabling easy integration with other tools or bots.
- **Interactive Dashboard (`Streamlit`):** A clean, user-friendly UI to view upcoming fixture predictions, model accuracy metrics, and expected probabilities.
- **Job Scheduler:** Automated daily updates for fetching fixtures and updating predictions.

## 🛠️ Tech Stack

- **Python 3.x**
- **Machine Learning:** `xgboost`, `scikit-learn`, `pandas`, `numpy`
- **Backend/API:** `FastAPI`, `uvicorn`, `SQLAlchemy`, `SQLite`
- **Frontend/Dashboard:** `Streamlit`
- **Web Scraping:** `BeautifulSoup`, `requests`

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/football_predictor.git
   cd football_predictor
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory. Add any necessary API keys or database configuration strings there (refer to `config/` if necessary).

## 🏃‍♂️ Usage

### Training the Models
Before generating predictions, you need to train the machine learning models.
```bash
python -m models.train
```

### Running the API Server
To start the FastAPI backend:
```bash
uvicorn api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can explore the interactive docs at `/docs`.

### Running the Streamlit Dashboard
To launch the interactive dashboard:
```bash
streamlit run dashboard/app.py
```

### Getting Betting Suggestions
To get a quick terminal output of the top recommended bets for today:
```bash
python suggest_bets.py
```

### Custom Match Predictions
If you want to run predictions for specific custom matches:
```bash
python predict_user_list.py
```

## 📂 Project Structure

```text
football_predictor/
│
├── api/                   # FastAPI backend implementation
├── config/                # Configuration settings & environment variables
├── dashboard/             # Streamlit dashboard UI
├── db/                    # Database models and SQLite setup
├── features/              # Feature engineering pipeline scripts
├── models/                # ML model definition and training scripts
├── scrapers/              # Scripts to fetch football fixture and result data
├── tests/                 # Unit testing suite
│
├── backfill.py            # Historical data backfilling utility
├── predict_advanced.py    # Advanced prediction logic
├── predict_custom.py      # Predict outcomes for individual inputs
├── predict_real.py        # Real-time prediction script
├── predict_user_list.py   # Batch process a user-provided list of matches
├── scheduler.py           # Automated job scheduling logic
├── suggest_bets.py        # Analyzes predictions to output recommended bets
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
