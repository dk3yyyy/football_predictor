import argparse
import logging
import streamlit as st
import pandas as pd
import json

from datetime import datetime, date
from sqlalchemy import text

# Add root project path when running from dashboard folder
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.database import Database

st.set_page_config(
    page_title="AI Football Predictor",
    page_icon="⚽",
    layout="wide"
)

# Connect DB safely for multithreading
@st.cache_resource
def get_db():
    return Database()

logger = logging.getLogger(__name__)

st.title("⚽ Advanced AI Football Match Predictor")

db = get_db()
query_metrics = """
    SELECT 
        COUNT(*) as total, 
        SUM(correct) as correct_preds 
    FROM predictions_log 
    WHERE actual_winner IS NOT NULL
"""
# Sidebar logic
st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("Use this panel to filter the models predictions by upcoming match dates or historical accuracy.")

try:
    with db.engine.connect() as conn:
        metrics = conn.execute(text(query_metrics)).fetchone()
        if metrics and metrics.total > 0:
            pct = metrics.correct_preds / metrics.total * 100
            st.sidebar.metric("Model Total Accuracy", f"{pct:.1f}%")
        else:
            st.sidebar.metric("Model Total Accuracy", "Pending Matches")
except Exception as e:
    st.sidebar.error("Database connection issue. Check configuration.")


# Main UI Tabs
tab1, tab2 = st.tabs(["Upcoming Predictions", "Match Details & Betting Value"])

with tab1:
    st.header("Upcoming Fixtures")
    date_filter = st.date_input("Filter by date:", date.today())
    
    query1 = """
        SELECT match_date, home_team, away_team, 
               pred_home_win, pred_draw, pred_away_win, 
               pred_home_goals, pred_away_goals
        FROM predictions_log 
        WHERE actual_winner IS NULL
        AND date(match_date) = date(:dt)
        ORDER BY match_date ASC
    """
    
    with db.engine.connect() as conn:
        df_upcoming = pd.read_sql(text(query1), conn, params={"dt": date_filter})
        
    if df_upcoming.empty:
        st.info(f"No predictions found for {date_filter}. Either wait for the scheduler to trigger or train the model if it's the first run.")
    else:
        # Create a visually appealing dataframe
        st.dataframe(
            df_upcoming.style.format({
                'pred_home_win': "{:.1%}",
                'pred_draw': "{:.1%}",
                'pred_away_win': "{:.1%}",
                'pred_home_goals': "{:.2f}",
                'pred_away_goals': "{:.2f}"
            }).background_gradient(subset=['pred_home_win'], cmap='Greens')
              .background_gradient(subset=['pred_away_win'], cmap='Reds')
        )

with tab2:
    st.header("Match Value Analysis")
    st.write("Compare the model probabilities against the bookmaker odds to find overlay value.")
    
    # Needs a join against the odds table (very advanced, using just predictions for now)
    query2 = """
        SELECT home_team, away_team, pred_home_win, pred_draw, pred_away_win
        FROM predictions_log
        WHERE actual_winner IS NULL
        ORDER BY match_date ASC LIMIT 5
    """
    
    with db.engine.connect() as conn:
        df_value = pd.read_sql(text(query2), conn)
        
    if not df_value.empty:
        for idx, row in df_value.iterrows():
            with st.expander(f"{row['home_team']} vs {row['away_team']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Home Win Prob", f"{row['pred_home_win']:.1%}")
                col2.metric("Draw Prob", f"{row['pred_draw']:.1%}")
                col3.metric("Away Win Prob", f"{row['pred_away_win']:.1%}")
                
                # Chart
                chart_data = pd.DataFrame(
                    [row['pred_home_win'], row['pred_draw'], row['pred_away_win']], 
                    index=["Home", "Draw", "Away"], 
                    columns=["Probability"]
                )
                st.bar_chart(chart_data, color="#2ecc71")
    else:
        st.write("Nothing to analyze.")
