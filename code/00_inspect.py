"""Raw-data checks before building sequences: frame rate, events, dir/o convention, joins."""
import duckdb
D = "/Users/ahanjain/bdb_ssl/data/raw"
con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='5GB'")
con.execute(f"CREATE VIEW gt AS SELECT *, year(\"time\") yr FROM '{D}/game_tracking_*.parquet'")
con.execute(f"CREATE VIEW pp AS SELECT * FROM '{D}/player_play.parquet'")
q = lambda s: print(con.execute(s).df().to_string(), "\n", flush=True)
q("SELECT count(*) n_frames, count(DISTINCT (game_id, play_id, nfl_id)) player_plays, count(DISTINCT nfl_id) players, count(DISTINCT game_id) games FROM gt")
q("SELECT event, count(*) n FROM gt GROUP BY 1 ORDER BY n DESC LIMIT 40")
q("""WITH p AS (SELECT game_id, play_id, nfl_id, count(*) nf, sum((event='ball_snap')::int) n_snap,
       epoch(max("time"))-epoch(min("time")) dur FROM gt GROUP BY ALL)
     SELECT quantile_cont(nf,[.01,.1,.5,.9,.99]) nf_q, avg((n_snap=1)::int) one_snap, avg((n_snap=0)::int) no_snap,
            quantile_cont(dur,[.01,.5,.99]) dur_q FROM p""")
q("""WITH s AS (SELECT game_id, play_id, nfl_id, min("time") FILTER (WHERE event='ball_snap') ts, min("time") t0, max("time") t1 FROM gt GROUP BY ALL)
     SELECT quantile_cont(epoch(ts)-epoch(t0),[.01,.1,.5,.9,.99]) pre_snap_s, quantile_cont(epoch(t1)-epoch(ts),[.01,.1,.5,.9,.99]) post_snap_s FROM s""")
q("""WITH f AS (SELECT s, dir, o, lead(x) OVER w - x dx, lead(y) OVER w - y dy,
       epoch(lead("time") OVER w) - epoch("time") dt
       FROM gt WHERE game_id % 23 = 0 WINDOW w AS (PARTITION BY game_id, play_id, nfl_id ORDER BY "time"))
     SELECT quantile_cont(dt,[.001,.01,.5,.99,.999]) dt_q,
       corr(dx/dt, s*sin(radians(dir))) c_vx_sin, corr(dy/dt, s*cos(radians(dir))) c_vy_cos,
       corr(dx/dt, s*cos(radians(dir))) c_vx_cos, corr(dy/dt, s*sin(radians(dir))) c_vy_sin,
       corr(sqrt(dx*dx+dy*dy)/dt, s) c_speed, avg((o IS NULL)::int) o_null, avg((dir IS NULL)::int) dir_null
     FROM f WHERE s > 1 AND dt BETWEEN 0.05 AND 0.15""")
# a signed or not?
q("SELECT avg((a<0)::int) frac_neg_a, min(a) amin, quantile_cont(a,[.01,.5,.99]) aq, min(s), max(s), min(x), max(x), min(y), max(y) FROM gt")
q("""SELECT count(*) trk, count(*) FILTER (WHERE p.game_id IS NOT NULL) AS n_matched FROM
     (SELECT DISTINCT game_id, play_id, nfl_id FROM gt) t LEFT JOIN pp p USING (game_id, play_id, nfl_id)""")
q("SELECT count(*) pp_rows, count(DISTINCT (game_id,play_id,nfl_id)) uniq, count(*) FILTER (WHERE play_direction IS NULL) no_dir FROM pp")
q(f"SELECT g.season, g.season_type, count(DISTINCT t.game_id) g FROM (SELECT DISTINCT game_id FROM gt) t JOIN '{D}/games.parquet' g USING (game_id) GROUP BY ALL ORDER BY ALL")
q("SELECT count(DISTINCT game_id) FILTER (WHERE game_id NOT IN (SELECT game_id FROM '" + D + "/games.parquet')) games_not_in_games_csv FROM gt")
q("SELECT lined_up_position, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("SELECT route_ran, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("SELECT coverage_assignment, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("SELECT team_coverage_type, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("SELECT offense_formation, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("SELECT receiver_alignment, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("SELECT pass_result, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("SELECT play_nullified_by_penalty, count(*) n FROM pp GROUP BY 1 ORDER BY n DESC")
q("""SELECT p.nfl_position, count(*) n, avg((pp.team_abbr=pp.possession_team)::int) off_share FROM pp JOIN '""" + D + """/players.parquet' p USING (nfl_id) GROUP BY 1 ORDER BY n DESC""")
q("SELECT count(*) FILTER (WHERE yardline_side IS NULL) ys_null, count(*) FILTER (WHERE yardline_number=50) at50 FROM pp")
