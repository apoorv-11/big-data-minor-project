"""
kafka_producer.py — Step 2 (streaming path): Replay transactions to Kafka.

Reads from the raw CSV (or the test-set slice) and publishes JSON messages
to the 'transactions' Kafka topic, simulating a live stream.

Usage:
  python -m src.ingestion.kafka_producer [--rate 100] [--limit 1000] [--start-step 601]
"""

import argparse
import csv
import json
import time

from confluent_kafka import Producer

from src.common.config import CONFIG
from src.common.logger import get_logger

logger = get_logger(__name__)


def delivery_report(err, msg):
    """Called once for each message produced to indicate delivery result."""
    if err is not None:
        logger.error("Delivery failed: %s", err)


def produce_transactions(rate: int, limit: int, start_step: int):
    """
    Read the CSV and send rows as JSON to Kafka.
    - rate: messages per second
    - limit: max messages (0 = unlimited)
    - start_step: only send rows with step >= this value
    """
    kafka_cfg = CONFIG["kafka"]
    topic = kafka_cfg["topic"]
    bootstrap = kafka_cfg["bootstrap_servers"]

    producer = Producer({"bootstrap.servers": bootstrap})
    logger.info("Kafka producer connected to %s, topic=%s", bootstrap, topic)
    logger.info("Rate=%d msg/s, limit=%d, start_step=%d", rate, limit, start_step)

    # Find the CSV
    import glob
    raw_dir = CONFIG["data"]["raw_dir"]
    csv_files = sorted(glob.glob(f"{raw_dir}/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {raw_dir}")
    csv_path = csv_files[0]
    logger.info("Reading from %s", csv_path)

    sent = 0
    delay = 1.0 / rate if rate > 0 else 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row.get("step", 0))
            if step < start_step:
                continue

            # Convert numeric fields
            msg = {
                "step": step,
                "type": row["type"],
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": row["nameDest"],
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "isFraud": int(row["isFraud"]),
                "isFlaggedFraud": int(row["isFlaggedFraud"]),
            }

            producer.produce(
                topic,
                key=row["nameOrig"],
                value=json.dumps(msg).encode("utf-8"),
                callback=delivery_report,
            )
            sent += 1

            if sent % 1000 == 0:
                producer.poll(0)  # trigger delivery reports
                logger.info("Sent %d messages (step=%d)", sent, step)

            if 0 < limit <= sent:
                break

            time.sleep(delay)

    producer.flush()
    logger.info("Finished: sent %d messages to topic '%s'", sent, topic)


def main():
    kafka_cfg = CONFIG["kafka"]["producer"]
    parser = argparse.ArgumentParser(description="Replay transactions to Kafka")
    parser.add_argument("--rate", type=int, default=kafka_cfg["rate"],
                        help="Messages per second")
    parser.add_argument("--limit", type=int, default=kafka_cfg["limit"],
                        help="Max messages (0=unlimited)")
    parser.add_argument("--start-step", type=int, default=kafka_cfg["start_step"],
                        help="Only send rows with step >= this")
    args = parser.parse_args()
    produce_transactions(args.rate, args.limit, args.start_step)


if __name__ == "__main__":
    main()
