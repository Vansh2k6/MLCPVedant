"""Historical windows via cumulative daily statistics and strict ASOF joins.

For each date t, the window is [t-window, t). No future row or same-day
appearance enters. Missing observations are not interpreted as injury/inactivity.
"""
from pathlib import Path
import duckdb

WINDOWS = (90, 180, 365)
MEASURES = {'appearances': 'COUNT(*)', 'minutes': 'SUM(minutes_played)',
            'goals': 'SUM(goals)', 'assists': 'SUM(assists)'}


def sql_path(path):
    return str(Path(path).resolve()).replace('\\', '/').replace("'", "''")


def build_recent_features(base_path, appearances_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        con.execute(f"CREATE VIEW base AS SELECT * FROM '{sql_path(base_path)}'")
        con.execute(f"CREATE VIEW appearances AS SELECT * FROM '{sql_path(appearances_path)}'")
        duplicates = con.execute('SELECT count(*)-count(DISTINCT appearance_id) FROM appearances').fetchone()[0]
        if duplicates:
            raise ValueError(f'{duplicates} duplicate appearance IDs; deduplicate at source first')
        duplicate_games = con.execute('SELECT count(*)-count(DISTINCT (player_id,game_id)) FROM appearances').fetchone()[0]
        if duplicate_games:
            raise ValueError(f'{duplicate_games} duplicate player-game records')
        if con.execute('SELECT count(*)-count(DISTINCT (player_id,valuation_date)) FROM base').fetchone()[0]:
            raise ValueError('Duplicate player-valuation rows')
        bad = con.execute('''SELECT count(*) FROM appearances WHERE date IS NULL OR player_id IS NULL
            OR minutes_played IS NULL OR goals IS NULL OR assists IS NULL
            OR minutes_played<0 OR goals<0 OR assists<0''').fetchone()[0]
        if bad:
            raise ValueError(f'{bad} invalid appearance records')
        daily = ', '.join(f'{expr} AS {name}' for name, expr in MEASURES.items())
        cumulative = ', '.join(f'SUM({name}) OVER w AS {name}' for name in MEASURES)
        con.execute(f'''CREATE TABLE cumulative AS WITH daily AS (
            SELECT player_id, CAST(date AS DATE) AS date, {daily}
            FROM appearances GROUP BY player_id, CAST(date AS DATE))
            SELECT player_id,date,{cumulative}, MIN(date) OVER w AS first_date FROM daily
            WINDOW w AS (PARTITION BY player_id ORDER BY date ROWS UNBOUNDED PRECEDING)''')
        columns, joins = [], []
        for days in WINDOWS:
            joins.append(f'''ASOF LEFT JOIN cumulative c{days} ON b.player_id=c{days}.player_id
                AND b.valuation_date-INTERVAL {days} DAY > c{days}.date''')
            for name in MEASURES:
                columns.append(f'COALESCE(a.{name},0)-COALESCE(c{days}.{name},0) AS observed_{name}_{days}d')
        con.execute(f'''COPY (SELECT b.*, {', '.join(columns)},
            a.date AS last_observed_appearance_date,
            datediff('day',a.date,b.valuation_date) AS days_since_last_appearance,
            datediff('day',a.first_date,b.valuation_date) AS observation_history_days,
            CASE WHEN a.date IS NULL THEN 0 ELSE 1 END AS has_observed_history
            FROM base b ASOF LEFT JOIN cumulative a
            ON b.player_id=a.player_id AND b.valuation_date>a.date
            {' '.join(joins)} ORDER BY b.valuation_date,b.player_id
            ) TO '{sql_path(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)''')
        # Cross-check independently recomputed annual stats against the original feature table.
        base_columns = {row[0] for row in con.execute('DESCRIBE base').fetchall()}
        if 'minutes_last_365d' in base_columns:
            mismatches = con.execute(f'''SELECT count(*) FROM '{sql_path(output_path)}'
                WHERE observed_minutes_365d != minutes_last_365d
                OR observed_goals_365d != goals_last_365d
                OR observed_assists_365d != assists_last_365d''').fetchone()[0]
            if mismatches:
                raise ValueError(f'{mismatches} annual-stat mismatches with base data; rebuild the base dataset')
    return output_path
