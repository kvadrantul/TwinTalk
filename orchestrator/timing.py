import random
import math
from typing import Optional


class TimingEngine:
    """
    Calculates realistic delays between messages based on actual response time
    distributions from the original chat history.
    """

    def __init__(self, response_times: list[float], speed_multiplier: float = 1.0):
        """
        response_times: list of actual response times in seconds from the chat history
        speed_multiplier: 0.5 = slower, 1.0 = normal, 2.0 = faster, 4.0 = very fast
        """
        self.response_times = sorted(response_times) if response_times else [10.0]
        self.speed_multiplier = speed_multiplier

    def get_next_delay(self) -> float:
        """
        Calculate the delay before the next message.

        Strategy:
        1. Sample from the actual response time distribution using percentile interpolation
        2. Add random jitter of ±20%
        3. Apply speed multiplier (divide by it)
        4. Clamp to [3.0, 60.0] seconds

        Returns delay in seconds.
        """
        sampled = self._sample_from_distribution()

        # Add ±20% jitter
        jitter = 1.0 + random.uniform(-0.2, 0.2)
        delay = sampled * jitter

        # Apply speed multiplier (higher = faster = shorter delay)
        delay = delay / self.speed_multiplier

        # Clamp to [3.0, 60.0]
        return max(3.0, min(60.0, delay))

    def set_speed_multiplier(self, multiplier: float):
        """Update speed multiplier (0.5, 1.0, 2.0, 4.0)."""
        self.speed_multiplier = multiplier

    def _sample_from_distribution(self) -> float:
        """
        Sample a value from the response time distribution.
        Uses random choice from the actual values, with interpolation between
        percentiles for smoother distribution coverage.
        """
        n = len(self.response_times)
        if n == 1:
            return self.response_times[0]

        # Pick a random percentile position (0.0 to 1.0)
        p = random.random()
        index = p * (n - 1)

        lower = math.floor(index)
        upper = math.ceil(index)

        if lower == upper:
            return self.response_times[lower]

        # Linear interpolation between adjacent values
        fraction = index - lower
        return self.response_times[lower] * (1 - fraction) + self.response_times[upper] * fraction
