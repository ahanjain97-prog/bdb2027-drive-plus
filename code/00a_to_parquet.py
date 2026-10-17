"""One-time CSV -> parquet conversion (typed, zstd) so later steps don't re-parse 2 GB of CSV."""
import duckdb, os
RAW = "/Users/ahanjain/Downloads/nfl-big-data-bowl-2027"
OUT = "/Users/ahanjain/bdb_ssl/data/raw"
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='5GB'")
for yr in (2023, 2024, 2025):
    con.execute(f"""COPY (SELECT game_id::BIGINT game_id, play_id::INT play_id, nfl_id::INT nfl_id,
        "time"::TIMESTAMP AS "time", x::FLOAT x, y::FLOAT y, s::FLOAT s, a::FLOAT a, dis::FLOAT dis,
        o::FLOAT o, dir::FLOAT dir, event
      FROM read_csv('{RAW}/game_tracking_{yr}.csv', header=true, nullstr='NA', auto_detect=true)
      ORDER BY game_id, play_id, nfl_id, "time")
      TO '{OUT}/game_tracking_{yr}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    print(yr, con.execute(f"SELECT count(*) FROM '{OUT}/game_tracking_{yr}.parquet'").fetchone())
con.execute(f"""COPY (SELECT * FROM read_csv('{RAW}/combine_tracking.csv', header=true, nullstr='NA', auto_detect=true))
  TO '{OUT}/combine_tracking.parquet' (FORMAT parquet, COMPRESSION zstd)""")
for f in ("player_play", "players", "combine_results", "games", "player_career_successes"):
    con.execute(f"""COPY (SELECT * FROM read_csv('{RAW}/{f}.csv', header=true, nullstr='NA', auto_detect=true, sample_size=-1))
      TO '{OUT}/{f}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    print(f, con.execute(f"SELECT count(*) FROM '{OUT}/{f}.parquet'").fetchone())
