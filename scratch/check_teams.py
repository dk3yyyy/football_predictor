from db.database import Database
from sqlalchemy import text
db = Database()
with db.engine.connect() as conn:
    teams = conn.execute(text("SELECT DISTINCT home_team_name FROM matches")).fetchall()
    for team in sorted([t[0] for t in teams]):
        print(team)
