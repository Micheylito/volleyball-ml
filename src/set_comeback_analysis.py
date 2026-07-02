from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live_set_db import load_set_comeback_rows


OUTPUT_DIR = Path("data/processed")
LEAD_THRESHOLDS = (4, 5, 6)


def prepare_set_records(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["set_number"] = pd.to_numeric(df["set_number"], errors="coerce")
    df["rally_number"] = pd.to_numeric(df["rally_number"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")
    df["set_winner"] = pd.to_numeric(df["set_winner"], errors="coerce")
    df["next_set_winner"] = pd.to_numeric(df["next_set_winner"], errors="coerce")
    df = df[
        df["set_number"].notna()
        & df["rally_number"].notna()
        & df["score1"].notna()
        & df["score2"].notna()
        & df["set_winner"].isin([1, 2])
    ].copy()
    df = df.sort_values(
        ["match_id", "set_number", "rally_number", "rally_ts"]
    ).reset_index(drop=True)

    records: list[dict[str, int | float]] = []
    for (match_id, set_number), group in df.groupby(["match_id", "set_number"], sort=False):
        max_team1_lead = float((group["score1"] - group["score2"]).max())
        max_team2_lead = float((group["score2"] - group["score1"]).max())
        set_winner = int(group["set_winner"].iloc[-1])
        next_set_winner_raw = group["next_set_winner"].iloc[-1]
        next_set_winner = int(next_set_winner_raw) if pd.notna(next_set_winner_raw) else None

        records.append(
            {
                "match_id": int(match_id),
                "set_number": int(set_number),
                "team": 1,
                "max_lead": max_team1_lead,
                "lost_set": int(set_winner != 1),
                "next_set_exists": int(next_set_winner in (1, 2)),
                "lost_next_set": int(next_set_winner == 2) if next_set_winner in (1, 2) else 0,
            }
        )
        records.append(
            {
                "match_id": int(match_id),
                "set_number": int(set_number),
                "team": 2,
                "max_lead": max_team2_lead,
                "lost_set": int(set_winner != 2),
                "next_set_exists": int(next_set_winner in (1, 2)),
                "lost_next_set": int(next_set_winner == 1) if next_set_winner in (1, 2) else 0,
            }
        )

    return pd.DataFrame(records)


def build_summary(records: pd.DataFrame, lead_thresholds: tuple[int, ...] = LEAD_THRESHOLDS) -> pd.DataFrame:
    all_next_sets = records[records["next_set_exists"] == 1].copy()
    control = all_next_sets[all_next_sets["lost_set"] == 1].copy()
    baseline_win_rate = float(1.0 - control["lost_next_set"].mean()) if not control.empty else 0.0

    summary_rows = [
        {
            "group": "all_set_losers_with_next_set",
            "samples": int(len(control)),
            "lose_next_set_rate": float(control["lost_next_set"].mean()) if not control.empty else 0.0,
            "win_next_set_rate": float(1.0 - control["lost_next_set"].mean())
            if not control.empty
            else 0.0,
            "avg_max_lead": float(control["max_lead"].mean()) if not control.empty else 0.0,
            "uplift_vs_baseline_win_rate": 0.0,
        },
    ]

    for lead_threshold in lead_thresholds:
        blown_lead = all_next_sets[
            (all_next_sets["max_lead"] >= lead_threshold) & (all_next_sets["lost_set"] == 1)
        ].copy()
        win_rate = float(1.0 - blown_lead["lost_next_set"].mean()) if not blown_lead.empty else 0.0
        summary_rows.append(
            {
                "group": f"blew_{lead_threshold}plus_lead_and_lost_set",
                "samples": int(len(blown_lead)),
                "lose_next_set_rate": float(blown_lead["lost_next_set"].mean())
                if not blown_lead.empty
                else 0.0,
                "win_next_set_rate": win_rate,
                "avg_max_lead": float(blown_lead["max_lead"].mean()) if not blown_lead.empty else 0.0,
                "uplift_vs_baseline_win_rate": win_rate - baseline_win_rate,
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    rows = load_set_comeback_rows()
    records = prepare_set_records(rows)
    summary = build_summary(records, LEAD_THRESHOLDS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records_path = OUTPUT_DIR / "set_comeback_records.csv"
    summary_path = OUTPUT_DIR / "set_comeback_summary.csv"
    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Set comeback analysis")
    print(f"Lead thresholds: {', '.join(str(value) + '+' for value in LEAD_THRESHOLDS)}")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.group}: samples={row.samples}, "
            f"lose_next_set_rate={row.lose_next_set_rate:.4f}, "
            f"win_next_set_rate={row.win_next_set_rate:.4f}, "
            f"avg_max_lead={row.avg_max_lead:.2f}, "
            f"uplift={row.uplift_vs_baseline_win_rate:.4f}"
        )
    print(f"Summary saved to {summary_path}")
    print(f"Records saved to {records_path}")


if __name__ == "__main__":
    main()
