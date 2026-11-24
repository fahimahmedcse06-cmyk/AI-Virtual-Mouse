# =============================================================================
# smoother.py — Optimized Cursor Position Smoother
# =============================================================================

import collections
import config

MAX_JUMP = 0.08


class Smoother:
    """Weighted moving average with velocity cap."""

    def __init__(self, buffer_size=None):
        size = buffer_size if buffer_size is not None else config.SMOOTHING_FRAMES

        if size < 1:
            raise ValueError("buffer_size must be at least 1")

        self._buf = collections.deque(maxlen=size)
        self._weights = tuple(range(1, size + 1))
        self._prev_x = None
        self._prev_y = None

        print(
            f"✅ Optimized Smoother ready → "
            f"buffer={size} frames, max_jump={MAX_JUMP}"
        )

    def smooth(self, x, y):
        # Limit sudden jumps
        if self._prev_x is not None:
            dx = x - self._prev_x
            dy = y - self._prev_y
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > MAX_JUMP:
                scale = MAX_JUMP / dist
                x = self._prev_x + dx * scale
                y = self._prev_y + dy * scale

        self._buf.append((x, y))
        self._prev_x = x
        self._prev_y = y

        # Weighted average without creating NumPy arrays
        count = len(self._buf)
        weights = self._weights[-count:]
        total_weight = sum(weights)

        smooth_x = sum(
            point[0] * weight
            for point, weight in zip(self._buf, weights)
        ) / total_weight

        smooth_y = sum(
            point[1] * weight
            for point, weight in zip(self._buf, weights)
        ) / total_weight

        return float(smooth_x), float(smooth_y)

    def reset(self):
        self._buf.clear()
        self._prev_x = None
        self._prev_y = None


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import random

    print("=" * 55)
    print("  OPTIMIZED SMOOTHER TEST")
    print("=" * 55)

    smoother = Smoother()

    base_x, base_y = 0.50, 0.50
    noise = 0.015

    print(f"\n{'Frame':>5}  {'Raw X':>7}  {'Raw Y':>7}  "
          f"{'Smooth X':>9}  {'Smooth Y':>9}")
    print("-" * 48)

    for i in range(20):
        raw_x = base_x + (i * 0.005) + random.uniform(-noise, noise)
        raw_y = base_y + random.uniform(-noise, noise)

        sx, sy = smoother.smooth(raw_x, raw_y)

        print(f"{i+1:>5}  {raw_x:>7.4f}  {raw_y:>7.4f}  "
              f"{sx:>9.4f}  {sy:>9.4f}")

    print("\n🔍 Velocity cap test:")

    test_smoother = Smoother(buffer_size=3)
    test_smoother.smooth(0.50, 0.50)

    result = test_smoother.smooth(0.95, 0.95)

    print(f"  Sudden jump result: ({result[0]:.3f}, {result[1]:.3f})")
    print(f"  Velocity cap working: {result[0] < 0.70}")

    print("\n✅ Optimized Smoother test complete.")
