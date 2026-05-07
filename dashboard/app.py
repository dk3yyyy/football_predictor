import logging
import os
import sys
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db.database import Database

st.set_page_config(
    page_title="AI Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Connect DB safely for multithreading
@st.cache_resource
def get_db():
    return Database()


logger = logging.getLogger(__name__)

db = get_db()

# Sidebar
st.sidebar.header("⚽ AI Football Predictor")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔮 Predictions", "📈 Model Performance", "✅ Results", "ℹ️ About"],
    index=0,
)

# Load data functions
@st.cache_data(ttl=300)
def load_metrics(live_only=True):
    query = """
        SELECT
            COUNT(*) as total,
            SUM(correct) as correct_preds,
            AVG(CASE WHEN actual_winner = 'HOME_TEAM' THEN pred_home_win
                     WHEN actual_winner = 'AWAY_TEAM' THEN pred_away_win
                     WHEN actual_winner = 'DRAW' THEN pred_draw END) as avg_prob
        FROM predictions_log
        WHERE actual_winner IS NOT NULL
    """
    if live_only:
        query += " AND model_name NOT LIKE '%backtest%'"

    with db.engine.connect() as conn:
        return conn.execute(text(query)).fetchone()


@st.cache_data(ttl=300)
def load_upcoming(date_filter):
    query = """
        SELECT DISTINCT match_id, match_date, home_team, away_team,
               pred_home_win, pred_draw, pred_away_win,
               pred_home_goals, pred_away_goals
        FROM predictions_log
        WHERE actual_winner IS NULL
        AND date(match_date) >= date(:dt)
        GROUP BY match_id
        ORDER BY match_date ASC
        LIMIT 50
    """
    with db.engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"dt": date_filter})
        # Calculate predicted winner from probabilities
        def get_winner(row):
            max_prob = max(row["pred_home_win"], row["pred_draw"], row["pred_away_win"])
            if max_prob == row["pred_home_win"]:
                return "HOME"
            elif max_prob == row["pred_away_win"]:
                return "AWAY"
            else:
                return "DRAW"
        df["predicted_winner"] = df.apply(get_winner, axis=1)
        return df


@st.cache_data(ttl=300)
def load_historical():
    query = """
        SELECT match_date, home_team, away_team, actual_winner, correct,
               pred_home_win, pred_draw, pred_away_win
        FROM predictions_log
        WHERE actual_winner IS NOT NULL
        ORDER BY match_date DESC
        LIMIT 100
    """
    with db.engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
        # Calculate predicted winner from probabilities
        def get_winner(row):
            max_prob = max(row["pred_home_win"], row["pred_draw"], row["pred_away_win"])
            if max_prob == row["pred_home_win"]:
                return "HOME"
            elif max_prob == row["pred_away_win"]:
                return "AWAY"
            else:
                return "DRAW"
        df["predicted_winner"] = df.apply(get_winner, axis=1)
        return df


@st.cache_data(ttl=300)
def load_feature_importance():
    try:
        import joblib
        from models.train import MODEL_FEATURES, GOAL_FEATURES

        model = joblib.load("models/outcome_ensemble.joblib")
        # CalibratedClassifierCV wraps the base model
        base_model = getattr(model, "estimator", model)
        if hasattr(base_model, "feature_importances_"):
            all_features = MODEL_FEATURES + GOAL_FEATURES
            importance = base_model.feature_importances_
            if len(importance) != len(all_features):
                all_features = [f"feat_{i}" for i in range(len(importance))]
            return pd.DataFrame(
                {"feature": all_features[: len(importance)], "importance": importance}
            ).sort_values("importance", ascending=False)
    except Exception:
        pass
    return None


if page == "📊 Dashboard":
    st.title("⚽ AI Football Predictor Dashboard")

    # Load metrics (live only, not backtest)
    metrics = load_metrics(live_only=True)

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if metrics and metrics.total > 0:
            accuracy = metrics.correct_preds / metrics.total * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%", delta=f"{accuracy - 50:.1f}% vs random")
        else:
            st.metric("Model Accuracy", "No data")

    with col2:
        if metrics and metrics.total > 0:
            st.metric("Total Predictions", f"{metrics.total}")
        else:
            st.metric("Total Predictions", "0")

    with col3:
        upcoming_query = "SELECT COUNT(DISTINCT match_id) as cnt FROM predictions_log WHERE actual_winner IS NULL"
        with db.engine.connect() as conn:
            result = conn.execute(text(upcoming_query)).fetchone()
            st.metric("Upcoming Matches", f"{result.cnt}" if result else "0")

    with col4:
        if metrics and metrics.avg_prob:
            st.metric("Avg Confidence", f"{metrics.avg_prob:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")

    st.markdown("---")

    # Main content
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🔮 Upcoming Predictions")

        date_filter = st.date_input("Filter from date:", date.today())
        df_upcoming = load_upcoming(date_filter)

        if df_upcoming.empty:
            st.info("No upcoming predictions found. Run the model trainer to generate predictions.")
        else:
            # Format dataframe for display
            display_df = df_upcoming.copy()
            display_df["match"] = display_df["home_team"] + " vs " + display_df["away_team"]
            display_df["confidence"] = display_df.apply(
                lambda r: max(r["pred_home_win"], r["pred_draw"], r["pred_away_win"]), axis=1
            )

            for _, row in display_df.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns([3, 2, 1])
                    with col_a:
                        st.markdown(f"**{row['home_team']} vs {row['away_team']}**")
                        st.caption(f"{row['match_date'].strftime('%Y-%m-%d %H:%M') if hasattr(row['match_date'], 'strftime') else row['match_date']}")
                    with col_b:
                        winner = row["predicted_winner"]
                        conf = row["confidence"]
                        if winner == "HOME":
                            st.success(f"🏠 Home Win ({conf:.1%})")
                        elif winner == "AWAY":
                            st.error(f"✈️ Away Win ({conf:.1%})")
                        else:
                            st.warning(f"🤝 Draw ({conf:.1%})")
                    with col_c:
                        st.metric("Pred Goals", f"{row['pred_home_goals']:.1f} - {row['pred_away_goals']:.1f}")
                st.divider()

    with col_right:
        st.subheader("📈 Model Insights")

        # Feature importance
        fi_df = load_feature_importance()
        if fi_df is not None:
            st.markdown("**Top 10 Important Features**")
            st.bar_chart(fi_df.head(10).set_index("feature")["importance"], color="#2ecc71")

        # Recent accuracy trend
        st.markdown("**Recent Performance**")
        hist_df = load_historical()
        if not hist_df.empty:
            hist_df["correct_int"] = hist_df["correct"].astype(int)
            hist_df["rolling_acc"] = hist_df["correct_int"].rolling(20, min_periods=1).mean()
            st.line_chart(hist_df.tail(50).set_index("match_date")["rolling_acc"], color="#3498db")

elif page == "🔮 Predictions":
    st.title("🔮 Match Predictions")

    tab1, tab2 = st.tabs(["Upcoming", "Historical"])

    with tab1:
        st.header("Upcoming Fixtures")
        date_filter = st.date_input("Show from:", date.today(), key="pred_date")

        df_upcoming = load_upcoming(date_filter)

        if df_upcoming.empty:
            st.info("No predictions available.")
        else:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                teams = ["All"] + sorted(pd.unique(df_upcoming[["home_team", "away_team"]].values.ravel()).tolist())
                selected_team = st.selectbox("Filter by team:", teams)

            with col2:
                min_conf = st.slider("Min confidence:", 0.0, 1.0, 0.0, 0.05)

            if selected_team != "All":
                df_upcoming = df_upcoming[
                    (df_upcoming["home_team"] == selected_team) | (df_upcoming["away_team"] == selected_team)
                ]

            # Display predictions
            for _, row in df_upcoming.iterrows():
                confidence = max(row["pred_home_win"], row["pred_draw"], row["pred_away_win"])
                if confidence < min_conf:
                    continue

                with st.expander(f"{row['home_team']} vs {row['away_team']} - {row['match_date'].strftime('%Y-%m-%d') if hasattr(row['match_date'], 'strftime') else row['match_date']}"):
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.metric("Home Win", f"{row['pred_home_win']:.1%}")
                        st.progress(row["pred_home_win"])

                    with col_b:
                        st.metric("Draw", f"{row['pred_draw']:.1%}")
                        st.progress(row["pred_draw"])

                    with col_c:
                        st.metric("Away Win", f"{row['pred_away_win']:.1%}")
                        st.progress(row["pred_away_win"])

                    st.caption(f"Predicted Score: {row['pred_home_goals']:.1f} - {row['pred_away_goals']:.1f}")

    with tab2:
        st.header("Historical Results")
        hist_df = load_historical()

        if hist_df.empty:
            st.info("No historical predictions yet.")
        else:
            # Summary stats
            total = len(hist_df)
            correct = hist_df["correct"].sum()
            accuracy = correct / total * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Matches", total)
            col2.metric("Correct Predictions", int(correct))
            col3.metric("Accuracy", f"{accuracy:.1f}%")

            # Detailed table
            st.dataframe(
                hist_df.style.apply(
                    lambda x: [
                        "background-color: #d4edda" if v else "background-color: #f8d7da"
                        for v in x["correct"]
                    ],
                    axis=1,
                ),
                use_container_width=True,
            )

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")

    # Load model artifacts
    try:
        import joblib

        outcome_model = joblib.load("models/outcome_ensemble.joblib")
        goals_model_home = joblib.load("models/goals_home_xgb.joblib")
        goals_model_away = joblib.load("models/goals_away_xgb.joblib")

        st.success("✅ Models loaded successfully")
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()

    # Model parameters
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Outcome Model (XGBoost)")
        if hasattr(outcome_model, "get_params"):
            params = outcome_model.get_params()
            st.json(
                {
                    "n_estimators": params.get("n_estimators"),
                    "max_depth": params.get("max_depth"),
                    "learning_rate": params.get("learning_rate"),
                    "objective": params.get("objective"),
                }
            )

    with col2:
        st.subheader("Goals Models (XGBoost)")
        st.caption("Home Goals Model")
        if hasattr(goals_model_home, "get_params"):
            params = goals_model_home.get_params()
            st.json(
                {
                    "n_estimators": params.get("n_estimators"),
                    "max_depth": params.get("max_depth"),
                    "learning_rate": params.get("learning_rate"),
                    "objective": params.get("objective"),
                }
            )

    st.markdown("---")

    # Feature importance visualization
    st.subheader("Feature Importance")
    fi_df = load_feature_importance()

    if fi_df is not None:
        tab1, tab2 = st.tabs(["Bar Chart", "Table"])

        with tab1:
            st.bar_chart(fi_df.set_index("feature")["importance"], color="#2ecc71")
            st.caption("Higher values indicate more influential features")

        with tab2:
            st.dataframe(fi_df.style.background_gradient(subset=["importance"], cmap="Greens"), use_container_width=True)
    else:
        st.warning("Feature importance data not available")

    st.markdown("---")

    # Historical performance
    st.subheader("Prediction Accuracy Over Time")
    hist_df = load_historical()

    if not hist_df.empty:
        hist_df["correct_int"] = hist_df["correct"].astype(int)
        hist_df["cumulative_acc"] = hist_df["correct_int"].expanding().mean()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cumulative Accuracy**")
            st.line_chart(hist_df.set_index("match_date")["cumulative_acc"], color="#2ecc71")

        with col2:
            st.markdown("**Recent Form (Last 50)**")
            hist_df["rolling_acc"] = hist_df["correct_int"].rolling(20, min_periods=1).mean()
            st.line_chart(hist_df.tail(50).set_index("match_date")["rolling_acc"], color="#3498db")

        # Distribution of predictions
        st.markdown("**Prediction Confidence Distribution**")
        fig_data = pd.DataFrame(
            {
                "Home Win": hist_df["pred_home_win"],
                "Draw": hist_df["pred_draw"],
                "Away Win": hist_df["pred_away_win"],
            }
        ).melt(var_name="Outcome", value_name="Probability")
        st.bar_chart(fig_data.groupby("Outcome")["Probability"].mean())

elif page == "✅ Results":
    st.title("✅ Prediction Results")
    
    # Filter options
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        show_type = st.radio("Show:", ["All", "Live Only", "Backtest Only"], horizontal=True)
    
    # Build query based on filter
    query = """
        SELECT match_date, home_team, away_team, 
               pred_home_win, pred_draw, pred_away_win,
               actual_winner, correct,
               model_name
        FROM predictions_log
        WHERE actual_winner IS NOT NULL
    """
    
    with db.engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    if df.empty:
        st.info("No completed predictions yet.")
    else:
        # Split into live vs backtest
        df["is_backtest"] = df["model_name"].str.contains("backtest", case=False)
        
        # Apply filter
        if show_type == "Live Only":
            df = df[~df["is_backtest"]]
        elif show_type == "Backtest Only":
            df = df[df["is_backtest"]]
        
        # Summary metrics
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        live_df = df[~df["is_backtest"]]
        backtest_df = df[df["is_backtest"]]
        
        with col1:
            total = len(df)
            if total > 0:
                acc = df["correct"].mean() * 100
                st.metric("Accuracy", f"{acc:.1f}%", delta=f"{acc - 50:.1f}% vs baseline")
            else:
                st.metric("Accuracy", "N/A")
        
        with col2:
            st.metric("Total", len(df))
        
        with col3:
            if not live_df.empty:
                live_acc = live_df["correct"].mean() * 100
                st.metric("Live", f"{live_acc:.1f}%")
            else:
                st.metric("Live", "0")
        
        with col4:
            if not backtest_df.empty:
                bt_acc = backtest_df["correct"].mean() * 100
                st.metric("Backtest", f"{bt_acc:.1f}%")
            else:
                st.metric("Backtest", "0")
        
        st.markdown("---")
        
        # Wins and Losses tabs
        wins_tab, losses_tab, all_tab = st.tabs(["✅ Wins", "❌ Losses", "📋 All Results"])
        
        with wins_tab:
            win_df = df[df["correct"] == 1].sort_values("match_date", ascending=False)
            st.subheader(f"Correct Predictions ({len(win_df)})")
            if not win_df.empty:
                # Color the actual_winner column
                styled = win_df[["match_date", "home_team", "away_team", "actual_winner", "pred_home_win", "pred_away_win"]].style.format({
                    "pred_home_win": "{:.1%}",
                    "pred_away_win": "{:.1%}"
                })
                st.dataframe(styled, use_container_width=True)
        
        with losses_tab:
            loss_df = df[df["correct"] == 0].sort_values("match_date", ascending=False)
            st.subheader(f"Incorrect Predictions ({len(loss_df)})")
            if not loss_df.empty:
                styled = loss_df[["match_date", "home_team", "away_team", "actual_winner", "pred_home_win", "pred_away_win"]].style.format({
                    "pred_home_win": "{:.1%}",
                    "pred_away_win": "{:.1%}"
                })
                st.dataframe(styled, use_container_width=True)
        
        with all_tab:
            st.subheader(f"All Results ({len(df)})")
            # Simple color mapping
            def color_correct(val):
                color = "#d4edda" if val == 1 else "#f8d7da"
                return f"background-color: {color}"

            st.dataframe(
                df[["match_date", "home_team", "away_team", "actual_winner", "correct", "is_backtest"]].style.apply(
                    color_correct, subset=["correct"]
                ),
                use_container_width=True
            )

elif page == "ℹ️ About":
    st.title("ℹ️ About AI Football Predictor")

    st.markdown(
        """
        ### How it works

        This system uses machine learning to predict football match outcomes:

        1. **Data Collection**: Fetches match data from Football-data.org API
        2. **Feature Engineering**: Calculates 21 features including:
           - Team form (overall and home/away split)
           - Rolling goals difference and points per game
           - Head-to-head records
           - Clean sheet and BTTS rates
           - Days of rest

        3. **Model Training**: Uses XGBoost for both outcome and goals prediction
        4. **Prediction**: Generates probabilities for Home Win, Draw, and Away Win

        ### Current Model Stats

        - **Features**: 27 engineered features (rolling stats + H2H + Elo + stacked goals)
        - **Algorithm**: XGBoost with isotonic calibration
        - **Training Data**: 3193 matches
        - **Accuracy**: ~56% (time-series validated)

        ### Limitations

        - No xG, possession, or detailed stats (empty DB tables)
        - No betting odds integration yet
        - Accuracy limited by available features (~59% achievable with current data)

        ### Technical Stack

        - **Backend**: Python, SQLAlchemy, XGBoost
        - **Database**: SQLite
        - **Frontend**: Streamlit
        - **Scheduling**: APScheduler
        """
    )

    st.markdown("---")
    st.caption("Built with ❤️ using AI • Data from Football-data.org")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
