"""Measure in-game calibration in the archived Kalshi and Polymarket trades.

The archive contains market-open timestamps, not reliable sports start times.
Game windows can come from ESPN historical play-by-play or from the existing
conservative market-activity window analysis.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import math
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import orjson
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET = "mm-predexon-archive-789877399631-us-east-1"
PREFIX = "predexon/v1"
PRICE_BINS = 100
TIME_BINS = 20
UTC = dt.UTC

_thread_local = threading.local()


@dataclass(frozen=True)
class Market:
    venue: str
    market_id: str
    game_key: str
    league: str
    s3_key: str
    outcomes: dict[str, int]


@dataclass
class Aggregate:
    count: np.ndarray
    wins: np.ndarray
    price_sum: np.ndarray
    volume: np.ndarray
    time_count: np.ndarray
    time_wins: np.ndarray
    time_price_sum: np.ndarray
    time_volume: np.ndarray
    total_rows: int = 0
    duplicate_rows: int = 0
    pregame_rows: int = 0
    postgame_rows: int = 0
    invalid_rows: int = 0
    ingame_rows: int = 0
    min_timestamp: int | None = None
    max_timestamp: int | None = None

    @classmethod
    def empty(cls) -> Aggregate:
        return cls(
            count=np.zeros(PRICE_BINS, dtype=np.int64),
            wins=np.zeros(PRICE_BINS, dtype=np.int64),
            price_sum=np.zeros(PRICE_BINS, dtype=np.float64),
            volume=np.zeros(PRICE_BINS, dtype=np.float64),
            time_count=np.zeros((TIME_BINS, PRICE_BINS), dtype=np.int64),
            time_wins=np.zeros((TIME_BINS, PRICE_BINS), dtype=np.int64),
            time_price_sum=np.zeros((TIME_BINS, PRICE_BINS), dtype=np.float64),
            time_volume=np.zeros((TIME_BINS, PRICE_BINS), dtype=np.float64),
        )

    def merge(self, other: Aggregate) -> None:
        self.count += other.count
        self.wins += other.wins
        self.price_sum += other.price_sum
        self.volume += other.volume
        self.time_count += other.time_count
        self.time_wins += other.time_wins
        self.time_price_sum += other.time_price_sum
        self.time_volume += other.time_volume
        self.total_rows += other.total_rows
        self.duplicate_rows += other.duplicate_rows
        self.pregame_rows += other.pregame_rows
        self.postgame_rows += other.postgame_rows
        self.invalid_rows += other.invalid_rows
        self.ingame_rows += other.ingame_rows
        if other.min_timestamp is not None:
            self.min_timestamp = (
                other.min_timestamp if self.min_timestamp is None else min(self.min_timestamp, other.min_timestamp)
            )
        if other.max_timestamp is not None:
            self.max_timestamp = (
                other.max_timestamp if self.max_timestamp is None else max(self.max_timestamp, other.max_timestamp)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/metrics/kalshi_polymarket_ingame_edge.json"),
    )
    parser.add_argument(
        "--schedule-cache",
        type=Path,
        default=Path("analysis/metrics/espn_game_windows.json"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--min-auc-bin-count", type=int, default=25)
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--limit-markets", type=int)
    parser.add_argument(
        "--existing-windows",
        type=Path,
        help="Optional JSONL game windows; restricts analysis to those common games",
    )
    return parser.parse_args()


def s3_client() -> Any:
    client = getattr(_thread_local, "s3", None)
    if client is None:
        client = boto3.client(
            "s3",
            config=Config(
                connect_timeout=10,
                read_timeout=120,
                retries={"max_attempts": 8, "mode": "adaptive"},
                max_pool_connections=64,
            ),
        )
        _thread_local.s3 = client
    return client


def read_s3_jsonl(key: str) -> list[dict[str, Any]]:
    body = s3_client().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return [orjson.loads(line) for line in body.splitlines() if line]


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def timestamp(value: str) -> int:
    return int(parse_utc(value).timestamp())


def game_key(row: dict[str, Any]) -> str | None:
    date = row.get("game_date")
    teams = row.get("teams")
    if not date or not teams or len(teams) != 2:
        return None
    return "|".join([row["league"], date, *sorted(teams)])


def load_markets() -> tuple[
    list[Market],
    dict[str, dict[str, Any]],
    dict[str, int],
    dict[str, str],
]:
    event_rows = read_s3_jsonl(f"{PREFIX}/manifests/event_map.jsonl")
    poly_manifest = read_s3_jsonl(f"{PREFIX}/manifests/polymarket_markets.jsonl")
    kalshi_manifest = read_s3_jsonl(f"{PREFIX}/manifests/kalshi_markets.jsonl")

    poly_events = {
        row["polymarket"]["condition_id"]: row
        for row in event_rows
        if row.get("polymarket") and row["polymarket"].get("condition_id")
    }
    kalshi_events = {
        row["kalshi"]["event_ticker"]: row
        for row in event_rows
        if row.get("kalshi") and row["kalshi"].get("event_ticker")
    }

    windows_needed: dict[str, dict[str, Any]] = {}
    markets: list[Market] = []
    exclusions: defaultdict[str, int] = defaultdict(int)

    for row in poly_manifest:
        source = row["source"]
        outcomes = source.get("outcomes", [])
        final_prices = [float(outcome.get("price", -1)) for outcome in outcomes]
        if len(outcomes) != 2 or sorted(final_prices) != [0.0, 1.0]:
            exclusions["polymarket_not_binary_resolved"] += 1
            continue
        event = poly_events.get(row["condition_id"])
        key = game_key(event) if event else None
        if key is None:
            exclusions["polymarket_missing_game_identity"] += 1
            continue
        token_outcomes = {
            str(outcome["token_id"]): int(float(outcome["price"]) == 1.0)
            for outcome in outcomes
            if outcome.get("token_id")
        }
        if len(token_outcomes) != 2:
            exclusions["polymarket_missing_tokens"] += 1
            continue
        windows_needed[key] = {
            "league": event["league"],
            "game_date": event["game_date"],
            "teams": sorted(event["teams"]),
        }
        markets.append(
            Market(
                venue="Polymarket",
                market_id=row["condition_id"],
                game_key=key,
                league=row["league"],
                s3_key=f"{PREFIX}/raw/trades/polymarket/{row['condition_id']}/pages.jsonl",
                outcomes=token_outcomes,
            )
        )

    for row in kalshi_manifest:
        source = row["source"]
        result = source.get("result")
        if result not in {"yes", "no"}:
            exclusions["kalshi_missing_explicit_result"] += 1
            continue
        event = kalshi_events.get(row["event_id"])
        key = game_key(event) if event else None
        if key is None:
            exclusions["kalshi_missing_game_identity"] += 1
            continue
        windows_needed[key] = {
            "league": event["league"],
            "game_date": event["game_date"],
            "teams": sorted(event["teams"]),
        }
        markets.append(
            Market(
                venue="Kalshi",
                market_id=row["market_id"],
                game_key=key,
                league=row["league"],
                s3_key=f"{PREFIX}/raw/trades/kalshi/{row['market_id']}/pages.jsonl",
                outcomes={row["market_id"]: int(result == "yes")},
            )
        )

    kalshi_event_keys = {
        event_ticker: key
        for event_ticker, row in kalshi_events.items()
        if (key := game_key(row)) is not None
    }
    return markets, windows_needed, dict(exclusions), kalshi_event_keys


def load_existing_windows(
    path: Path,
    kalshi_event_keys: dict[str, str],
) -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    for line in path.read_bytes().splitlines():
        row = orjson.loads(line)
        key = kalshi_event_keys.get(row.get("kalshi_event", ""))
        if key is None or "game_start_ts" not in row or "game_end_ts" not in row:
            continue
        start = int(row["game_start_ts"])
        end = int(row["game_end_ts"])
        if end <= start:
            continue
        windows[key] = {
            "league": row["league"],
            "game_date": row["game_date"],
            "teams": key.split("|")[2:],
            "start": start,
            "start_iso": dt.datetime.fromtimestamp(start, UTC).isoformat().replace("+00:00", "Z"),
            "end": end,
            "end_iso": dt.datetime.fromtimestamp(end, UTC).isoformat().replace("+00:00", "Z"),
            "end_source": "last non-pinned Polymarket minute candle",
            "window_method": row.get("window_method", "unknown"),
            "duration_minutes": round((end - start) / 60, 2),
        }
    return windows


def canonical_team(league: str, abbreviation: str) -> str:
    aliases = {
        "NBA": {
            "GS": "GSW",
            "NO": "NOP",
            "NY": "NYK",
            "SA": "SAS",
            "UTAH": "UTA",
            "WSH": "WAS",
        },
        "NFL": {
            "JAX": "JAC",
            "WSH": "WAS",
        },
    }
    normalized = abbreviation.upper().replace(" ", "")
    return aliases.get(league, {}).get(normalized, normalized)


def espn_sport(league: str) -> str:
    return "basketball/nba" if league == "NBA" else "football/nfl"


def request_json(url: str, attempts: int = 5) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "market-calibration-research/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return orjson.loads(response.read())
        except (urllib.error.URLError, TimeoutError, orjson.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError("unreachable")


def fetch_scoreboard(item: tuple[str, str]) -> tuple[tuple[str, str], list[dict[str, Any]]]:
    league, date = item
    compact_date = date.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport(league)}/scoreboard"
    data = request_json(f"{url}?dates={compact_date}&limit=100")
    return item, data.get("events", [])


def extract_final_play(summary: dict[str, Any], start: int) -> tuple[int | None, str | None]:
    wallclocks: list[int] = []
    end_game_wallclocks: list[int] = []
    for play in summary.get("plays", []):
        wallclock = play.get("wallclock")
        if not wallclock:
            continue
        try:
            value = timestamp(wallclock)
        except (TypeError, ValueError):
            continue
        if value < start:
            continue
        wallclocks.append(value)
        play_type = str(play.get("type", {}).get("text", "")).lower()
        play_text = str(play.get("text", "")).lower()
        if "end game" in play_type or "end of game" in play_text or "game end" in play_text:
            end_game_wallclocks.append(value)
    if end_game_wallclocks:
        return max(end_game_wallclocks), "ESPN final-play wallclock"
    if wallclocks:
        return max(wallclocks), "ESPN last-play wallclock"
    return None, None


def fetch_summary(
    item: tuple[str, dict[str, Any]],
) -> tuple[str, dict[str, Any] | None, str | None]:
    key, window = item
    url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport(window['league'])}/summary"
    try:
        summary = request_json(f"{url}?event={window['espn_event_id']}")
    except Exception as exc:  # noqa: BLE001 - preserve the game-level failure and continue
        return key, None, f"{type(exc).__name__}: {exc}"
    end, source = extract_final_play(summary, window["start"])
    if end is None:
        return key, None, "no play wallclock"
    duration = end - window["start"]
    if duration < 30 * 60 or duration > 12 * 60 * 60:
        return key, None, f"implausible duration: {duration}s"
    completed = dict(window)
    completed["end"] = end
    completed["end_iso"] = dt.datetime.fromtimestamp(end, UTC).isoformat().replace("+00:00", "Z")
    completed["end_source"] = source
    completed["duration_minutes"] = round(duration / 60, 2)
    return key, completed, None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temp.replace(path)


def build_game_windows(
    needed: dict[str, dict[str, Any]],
    cache_path: Path,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cached_payload = orjson.loads(cache_path.read_bytes())
        cached = cached_payload.get("windows", {})
    complete = {key: value for key, value in cached.items() if key in needed and value.get("end")}
    missing = {key: value for key, value in needed.items() if key not in complete}
    if not missing:
        return complete, {}

    dates = sorted({(value["league"], value["game_date"]) for value in missing.values()})
    scoreboards: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 20)) as executor:
        futures = [executor.submit(fetch_scoreboard, item) for item in dates]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            item, events = future.result()
            scoreboards[item] = events
            if index % 100 == 0 or index == len(futures):
                print(f"schedule scoreboards: {index}/{len(futures)}", flush=True)

    event_index: defaultdict[tuple[str, str, tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for (league, requested_date), events in scoreboards.items():
        for event in events:
            competitors = event.get("competitions", [{}])[0].get("competitors", [])
            teams = tuple(
                sorted(
                    canonical_team(league, competitor.get("team", {}).get("abbreviation", ""))
                    for competitor in competitors
                )
            )
            if len(teams) != 2 or not all(teams):
                continue
            event_index[(league, requested_date, teams)].append(event)

    matched: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for key, target in missing.items():
        index_key = (target["league"], target["game_date"], tuple(target["teams"]))
        candidates = event_index.get(index_key, [])
        if len(candidates) != 1:
            failures[key] = f"scoreboard matches={len(candidates)}"
            continue
        event = candidates[0]
        start = timestamp(event["date"])
        matched[key] = {
            **target,
            "espn_event_id": event["id"],
            "event_name": event.get("name"),
            "start": start,
            "start_iso": dt.datetime.fromtimestamp(start, UTC).isoformat().replace("+00:00", "Z"),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 20)) as executor:
        futures = [executor.submit(fetch_summary, item) for item in matched.items()]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key, window, error = future.result()
            if window is not None:
                complete[key] = window
            else:
                failures[key] = error or "unknown summary failure"
            if index % 100 == 0 or index == len(futures):
                print(f"schedule summaries: {index}/{len(futures)}", flush=True)

    payload = {
        "generated_at": dt.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": "ESPN historical scoreboard and game-summary APIs",
        "windows": complete,
        "failures": failures,
    }
    write_json(cache_path, payload)
    return complete, failures


def price_bin(price: float) -> int:
    return min(PRICE_BINS - 1, max(0, math.floor(price * PRICE_BINS)))


def progress_bin(trade_time: int, start: int, end: int) -> int:
    progress = (trade_time - start) / (end - start)
    return min(TIME_BINS - 1, max(0, math.floor(progress * TIME_BINS)))


def polymarket_identity(trade: dict[str, Any]) -> tuple[Any, ...]:
    return (
        trade.get("tx_hash"),
        trade.get("order_hash"),
        trade.get("token_id"),
        trade.get("side"),
        trade.get("user"),
        trade.get("shares"),
        trade.get("price"),
    )


def iter_trade_pages(key: str) -> Iterable[list[dict[str, Any]]]:
    response = s3_client().get_object(Bucket=BUCKET, Key=key)
    for line in response["Body"].iter_lines(chunk_size=1024 * 1024):
        if not line:
            continue
        page = orjson.loads(line)
        yield page.get("response", {}).get("trades", [])


def process_market(market: Market, window: dict[str, Any]) -> tuple[Market, Aggregate, str | None]:
    aggregate = Aggregate.empty()
    seen: set[Any] = set()
    start = int(window["start"])
    end = int(window["end"])
    try:
        pages = iter_trade_pages(market.s3_key)
        for trades in pages:
            for trade in trades:
                aggregate.total_rows += 1
                if market.venue == "Kalshi":
                    identity = trade.get("trade_id")
                    trade_time = int(trade.get("created_time", 0))
                    outcome_key = market.market_id
                    raw_price = trade.get("yes_price")
                    raw_volume = trade.get("count", 0)
                else:
                    identity = polymarket_identity(trade)
                    trade_time = int(trade.get("timestamp", 0))
                    outcome_key = str(trade.get("token_id", ""))
                    raw_price = trade.get("price")
                    raw_volume = trade.get("amount_usd", 0)

                if identity in seen:
                    aggregate.duplicate_rows += 1
                    continue
                seen.add(identity)

                outcome = market.outcomes.get(outcome_key)
                try:
                    price = float(raw_price)
                    volume = float(raw_volume)
                except (TypeError, ValueError):
                    aggregate.invalid_rows += 1
                    continue
                if outcome is None or not math.isfinite(price) or price < 0 or price > 1 or trade_time <= 0:
                    aggregate.invalid_rows += 1
                    continue
                if trade_time < start:
                    aggregate.pregame_rows += 1
                    continue
                if trade_time > end:
                    aggregate.postgame_rows += 1
                    continue

                p_bin = price_bin(price)
                t_bin = progress_bin(trade_time, start, end)
                aggregate.count[p_bin] += 1
                aggregate.wins[p_bin] += outcome
                aggregate.price_sum[p_bin] += price
                aggregate.volume[p_bin] += volume
                aggregate.time_count[t_bin, p_bin] += 1
                aggregate.time_wins[t_bin, p_bin] += outcome
                aggregate.time_price_sum[t_bin, p_bin] += price
                aggregate.time_volume[t_bin, p_bin] += volume
                aggregate.ingame_rows += 1
                aggregate.min_timestamp = (
                    trade_time if aggregate.min_timestamp is None else min(aggregate.min_timestamp, trade_time)
                )
                aggregate.max_timestamp = (
                    trade_time if aggregate.max_timestamp is None else max(aggregate.max_timestamp, trade_time)
                )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "ClientError")
        return market, aggregate, code
    except Exception as exc:  # noqa: BLE001 - record a market-level failure without aborting the full scan
        return market, aggregate, f"{type(exc).__name__}: {exc}"
    return market, aggregate, None


def approximate_roc_auc(count: np.ndarray, wins: np.ndarray) -> float | None:
    positives = int(wins.sum())
    negatives = int((count - wins).sum())
    if positives == 0 or negatives == 0:
        return None
    favorable = 0.0
    negatives_below = 0
    for total, positive in zip(count, wins, strict=True):
        negative = int(total - positive)
        favorable += int(positive) * (negatives_below + 0.5 * negative)
        negatives_below += negative
    return favorable / (positives * negatives)


def calibration_points(
    count: np.ndarray,
    wins: np.ndarray,
    price_sum: np.ndarray,
    volume: np.ndarray,
) -> list[dict[str, Any]]:
    points = []
    for index in range(PRICE_BINS):
        n = int(count[index])
        if n == 0:
            continue
        mean_price = float(price_sum[index] / n)
        resolved_rate = float(wins[index] / n)
        points.append(
            {
                "bucket": index,
                "bucket_label": f"{index}-{index + 1}%",
                "mean_price_pct": round(mean_price * 100, 6),
                "resolved_pct": round(resolved_rate * 100, 6),
                "edge_pp": round((resolved_rate - mean_price) * 100, 6),
                "trades": n,
                "volume": round(float(volume[index]), 6),
            }
        )
    return points


def calibration_auc(
    count: np.ndarray,
    wins: np.ndarray,
    price_sum: np.ndarray,
    min_bin_count: int,
) -> dict[str, Any]:
    valid = count >= min_bin_count
    valid_bins = int(valid.sum())
    if valid_bins < 2:
        return {
            "absolute_calibration_auc_pp": None,
            "signed_calibration_auc_pp": None,
            "roc_auc": approximate_roc_auc(count, wins),
            "price_bins_used": valid_bins,
        }
    x = price_sum[valid] / count[valid]
    y = wins[valid] / count[valid]
    order = np.argsort(x)
    x = x[order]
    gap = y[order] - x
    x_full = np.concatenate(([0.0], x, [1.0]))
    signed_gap = np.concatenate(([0.0], gap, [0.0]))
    absolute_auc = float(np.trapezoid(np.abs(signed_gap), x_full) * 100)
    signed_auc = float(np.trapezoid(signed_gap, x_full) * 100)
    return {
        "absolute_calibration_auc_pp": round(absolute_auc, 6),
        "signed_calibration_auc_pp": round(signed_auc, 6),
        "roc_auc": round(float(approximate_roc_auc(count, wins) or 0), 6),
        "price_bins_used": valid_bins,
    }


def summarize_scope(aggregate: Aggregate, min_auc_bin_count: int) -> dict[str, Any]:
    points = calibration_points(aggregate.count, aggregate.wins, aggregate.price_sum, aggregate.volume)
    total = int(aggregate.count.sum())
    total_wins = int(aggregate.wins.sum())
    total_price = float(aggregate.price_sum.sum())
    weighted_abs_gap = (
        sum(point["trades"] * abs(point["edge_pp"]) for point in points) / total if total else None
    )
    time_metrics = []
    for index in range(TIME_BINS):
        metrics = calibration_auc(
            aggregate.time_count[index],
            aggregate.time_wins[index],
            aggregate.time_price_sum[index],
            min_auc_bin_count,
        )
        n = int(aggregate.time_count[index].sum())
        win_count = int(aggregate.time_wins[index].sum())
        price_total = float(aggregate.time_price_sum[index].sum())
        time_metrics.append(
            {
                "progress_start_pct": index * 100 // TIME_BINS,
                "progress_end_pct": (index + 1) * 100 // TIME_BINS,
                "progress_midpoint_pct": (index + 0.5) * 100 / TIME_BINS,
                "trades": n,
                "mean_signed_edge_pp": round((win_count - price_total) / n * 100, 6) if n else None,
                **metrics,
            }
        )
    overall_auc = calibration_auc(
        aggregate.count,
        aggregate.wins,
        aggregate.price_sum,
        min_auc_bin_count,
    )
    return {
        "trade_rows_scanned": aggregate.total_rows,
        "duplicate_rows_removed": aggregate.duplicate_rows,
        "pregame_rows_excluded": aggregate.pregame_rows,
        "postgame_rows_excluded": aggregate.postgame_rows,
        "invalid_rows_excluded": aggregate.invalid_rows,
        "ingame_trades": aggregate.ingame_rows,
        "first_ingame_trade": (
            dt.datetime.fromtimestamp(aggregate.min_timestamp, UTC).isoformat().replace("+00:00", "Z")
            if aggregate.min_timestamp
            else None
        ),
        "last_ingame_trade": (
            dt.datetime.fromtimestamp(aggregate.max_timestamp, UTC).isoformat().replace("+00:00", "Z")
            if aggregate.max_timestamp
            else None
        ),
        "mean_signed_edge_pp": round((total_wins - total_price) / total * 100, 6) if total else None,
        "trade_weighted_absolute_bucket_gap_pp": round(weighted_abs_gap, 6) if weighted_abs_gap is not None else None,
        **overall_auc,
        "calibration": points,
        "time": time_metrics,
    }


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    markets, needed_windows, manifest_exclusions, kalshi_event_keys = load_markets()
    eligible_before_window_filter = len(markets)
    window_source: str
    if args.existing_windows:
        windows = load_existing_windows(args.existing_windows, kalshi_event_keys)
        needed_windows = {key: value for key, value in needed_windows.items() if key in windows}
        markets = [market for market in markets if market.game_key in windows]
        window_failures: dict[str, str] = {}
        window_source = (
            "Archived Polymarket candle activity: continuous-trading onset or a conservative nominal-duration "
            "fallback through the last non-pinned minute"
        )
    else:
        windows = {}
        window_failures = {}
        window_source = "ESPN historical schedule and final-play wallclocks"
    window_population_exclusions = eligible_before_window_filter - len(markets)
    if args.limit_markets:
        markets = markets[: args.limit_markets]
        needed_keys = {market.game_key for market in markets}
        needed_windows = {key: value for key, value in needed_windows.items() if key in needed_keys}
        windows = {key: value for key, value in windows.items() if key in needed_keys}
    print(
        f"eligible markets: {len(markets)}; unique games needing windows: {len(needed_windows)}",
        flush=True,
    )

    if not args.existing_windows:
        windows, window_failures = build_game_windows(needed_windows, args.schedule_cache, args.workers)
    print(f"complete game windows: {len(windows)}; failures: {len(window_failures)}", flush=True)
    if args.schedule_only:
        return

    included_markets = [market for market in markets if market.game_key in windows]
    market_window_exclusions = window_population_exclusions + len(markets) - len(included_markets)
    if args.limit_markets:
        included_markets = included_markets[: args.limit_markets]

    aggregates: defaultdict[tuple[str, str], Aggregate] = defaultdict(Aggregate.empty)
    games_by_scope: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    markets_by_scope: defaultdict[tuple[str, str], int] = defaultdict(int)
    scan_failures: dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_market, market, windows[market.game_key])
            for market in included_markets
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            market, aggregate, error = future.result()
            if error:
                scan_failures[f"{market.venue}:{market.market_id}"] = error
            else:
                for league in ("All", market.league):
                    scope = (market.venue, league)
                    aggregates[scope].merge(aggregate)
                    games_by_scope[scope].add(market.game_key)
                    markets_by_scope[scope] += 1
            if index % 100 == 0 or index == len(futures):
                rows = sum(value.total_rows for key, value in aggregates.items() if key[1] == "All")
                ingame = sum(value.ingame_rows for key, value in aggregates.items() if key[1] == "All")
                print(
                    f"trade files: {index}/{len(futures)}; rows={rows:,}; in-game={ingame:,}",
                    flush=True,
                )

    scopes: dict[str, dict[str, Any]] = {}
    for (venue, league), aggregate in sorted(aggregates.items()):
        name = f"{venue}:{league}"
        scopes[name] = {
            "venue": venue,
            "league": league,
            "markets": markets_by_scope[(venue, league)],
            "games": len(games_by_scope[(venue, league)]),
            **summarize_scope(aggregate, args.min_auc_bin_count),
        }

    durations: defaultdict[str, list[float]] = defaultdict(list)
    for window in windows.values():
        durations[window["league"]].append(float(window["duration_minutes"]))
    duration_summary = {
        league: {
            "games": len(values),
            "median_minutes": round(float(np.median(values)), 2),
            "p10_minutes": round(float(np.percentile(values, 10)), 2),
            "p90_minutes": round(float(np.percentile(values, 90)), 2),
        }
        for league, values in sorted(durations.items())
    }

    output = {
        "generated_at": dt.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runtime_seconds": round(time.monotonic() - started, 2),
        "source": f"s3://{BUCKET}/{PREFIX}/raw/trades",
        "methodology": {
            "population": (
                "Resolved NBA and NFL winner-market token/contract trade records"
                + (" on games archived at both venues" if args.existing_windows else "")
            ),
            "in_game_window": window_source,
            "price_buckets": "100 half-open 1 percentage-point bins; 100% is included in 99-100%",
            "weighting": "Each deduplicated API trade row receives equal weight",
            "kalshi_price": "yes_price; one-cent native price grid; created_time is seconds, created_time_ms is ms",
            "polymarket_price": "Outcome-token price; second-resolution timestamp; both BUY and SELL records included",
            "resolution": (
                "Kalshi requires explicit result=yes/no; Polymarket requires final outcome-token prices exactly 1/0"
            ),
            "absolute_calibration_auc": (
                "Integral over probability of abs(empirical resolution rate - mean traded price), in percentage points; "
                "empty endpoints are anchored to zero gap"
            ),
            "signed_calibration_auc": (
                "Integral over probability of empirical resolution rate - mean traded price, in percentage points"
            ),
            "roc_auc": "Approximate trade-level ROC AUC from the 1% price histograms",
            "time_axis": (
                "20 equal bins of the conservative market-activity game window"
                if args.existing_windows
                else "20 equal bins of actual wall-clock game progress from scheduled start to final play"
            ),
            "min_trades_per_auc_price_bin": args.min_auc_bin_count,
        },
        "granularity": {
            "Kalshi": {"timestamp": "milliseconds available; seconds used for the join", "price": "1 cent"},
            "Polymarket": {"timestamp": "1 second", "price": "continuous floating-point token price"},
        },
        "archive_manifest_exclusions": manifest_exclusions,
        "eligible_markets_before_game_window_filter": eligible_before_window_filter,
        "market_window_exclusions": market_window_exclusions,
        "game_window_failures": window_failures,
        "trade_file_failures": scan_failures,
        "game_duration_summary": duration_summary,
        "scopes": scopes,
        "limitations": [
            "Calibration is not executable arbitrage: spread, fees, latency, slippage, and available size are excluded.",
            "Trade rows from the same game are correlated; raw trade counts are not independent samples.",
            "Polymarket records are outcome-token-side rows, so a blockchain transaction can contribute multiple rows.",
            (
                "The common-game window is conservative and market-derived; it is not exact scoreboard game time."
                if args.existing_windows
                else "Only games with an ESPN match and a plausible final-play wallclock are included."
            ),
        ],
    }
    write_json(args.output, output)
    print(f"wrote {args.output} in {output['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
