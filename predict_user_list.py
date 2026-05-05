import os
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress XGBoost warnings
warnings.filterwarnings('ignore')

from predict_custom import predict_custom_match

matches = [
    ("Inter", "Cagliari"),
    ("FC Dynamo Kyiv", "FC Zorya Luhansk"),
    ("Vitesse Arnhem", "MVV Maastricht"),
    ("Al Ahli Saudi FC", "Johor Darul Tazim FC"),
    ("Fenerbahce Istanbul", "Caykur Rizespor"),
    ("FK Rostov", "FK Sochi"),
    ("SV 07 Elversberg", "Karlsruher SC"),
    ("Roda JC Kerkrade", "FC Emmen"),
    ("RC Lens", "Toulouse"),
    ("Sassuolo", "Como")
]

if __name__ == "__main__":
    print("🚀 Starting Batch Prediction for User-Requested Matches...\n")
    for home, away in matches:
        predict_custom_match(home, away)
    print("\n✅ Batch Prediction Complete.")
