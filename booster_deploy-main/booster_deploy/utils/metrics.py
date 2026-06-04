from __future__ import annotations

import time
import logging

import numpy as np

from .synced_array import SyncedArray

logger = logging.getLogger("booster_metrics")


class SyncedMetrics:
    """Lightweight cross-process event timestamp recorder using SyncedArray

    Layout:
        buf[0] = write_pos (next write slot, 0..max_events-1)
        buf[1] = total_written (monotonic count)
        buf[2:2+max_events] = timestamps (float64)

    When full, the buffer overwrites the oldest data in a ring.
    """

    def __init__(self, name: str, max_events: int = 10000):
        self.name = name
        self.max_events = int(max_events)
        # allocate shared array: 2 control slots + max_events timestamps
        self._arr = SyncedArray(
            f"metric_{name}",
            shape=(self.max_events + 2,),
            dtype="float64",
        )

    def mark(self) -> None:
        """Record the current timestamp into the shared buffer atomically
        (uses `modify_in_place`)."""
        def _updater(buf: np.ndarray):
            # buf layout: [write_pos, total_written, t0, t1, ...]
            write_pos = int(buf[0])
            total = int(buf[1])
            buf[2 + write_pos] = time.perf_counter()
            write_pos = (write_pos + 1) % self.max_events
            buf[0] = float(write_pos)
            buf[1] = float(total + 1)

        self._arr.modify_in_place(_updater)

    def compute(self):
        """Read the shared buffer and return statistics:
        count, freq_hz, mean_period_s, min_period_s, max_period_s."""
        data = self._arr.read()
        write_pos = int(data[0])
        total = int(data[1])
        if total < 2:
            return {
                "count": int(total),
                "freq_hz": 0.0,
                "mean_period_s": None,
                "min_period_s": None,
                "max_period_s": None,
            }

        if total < self.max_events:
            ts = data[2:2 + total]
        else:
            # buffer full: oldest at write_pos
            if write_pos == 0:
                ts = data[2:2 + self.max_events]
            else:
                part1 = data[2 + write_pos:2 + self.max_events]
                part2 = data[2:2 + write_pos]
                ts = np.concatenate([part1, part2])

        periods = np.diff(ts)
        mean_p = float(np.mean(periods))
        return {
            "count": int(min(total, self.max_events)),
            "freq_hz": 1.0 / mean_p if mean_p > 0 else float("inf"),
            "mean_period_s": mean_p,
            "min_period_s": float(np.min(periods)),
            "max_period_s": float(np.max(periods)),
        }

    def cleanup(self) -> None:
        try:
            self._arr.cleanup()
        except Exception:
            pass


__all__ = ["SyncedMetrics"]
