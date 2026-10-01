import math
from dataclasses import dataclass


@dataclass
class FTDDStats:
    p_value: float = 1.0
    drift_count: int = 0
    warning_count: int = 0
    error_rate_reference: float = 0.0
    error_rate_current: float = 0.0


class FTDDDriftDetector:

    def __init__(
        self,
        window_size: int = 50,
        drift_level: float = 0.05,
        warning_level: float = 0.1,
    ):
        self.w = window_size    # w  (recent window size)
        self.alpha_d = drift_level   # αd
        self.alpha_w = warning_level  # αw

        # Algorithm 1, lines 3-5: precompute factorials and constF
        self.maxim = 2 * self.w
        self.fact = [math.factorial(i) for i in range(self.maxim + 1)]
        self.const_f = (self.fact[self.w] ** 2) / self.fact[self.maxim]

        self.stats = FTDDStats()
        self._reset_method_stats()

    def _reset_method_stats(self) -> None:
        """Algorithm 1, line 1 / line 8: reset methodStats."""
        self.drift_detected = False
        self.is_warning_zone = False
        self.no = 0   # total samples in older window
        self.wo = 0   # wrong predictions in older window
        self.recent: list = []  # recent window, length <= w

    def _compute_p_value(self, wr: int, wp: int) -> float:
        """Algorithm 4: FTDD p-value calculation."""
        rr = self.w - wr
        rp = self.w - wp
        fact = self.fact
        p = (fact[wr + wp] / fact[wr] / fact[wp] *
             fact[rr + rp] / fact[rr] / fact[rp])
        return min(p * self.const_f * 2, 1.0)

    def get_current_stats(self) -> FTDDStats:
        return self.stats

    def update(self, prediction, true_label) -> str:
        """
        Update detector with one sample.
        Returns 'drift', 'warning', or 'normal'.
        """
        error = 1 if prediction != true_label else 0

        # Algorithm 1, lines 7-9: reset at start of next instance after drift
        if self.drift_detected:
            self._reset_method_stats()

        # Algorithm 1, line 10: update older and recent windows
        # When recent window is full, oldest element moves to older window
        if len(self.recent) == self.w:
            oldest = self.recent.pop(0)
            self.no += 1
            self.wo += oldest
        self.recent.append(error)

        # Algorithm 1, line 11
        self.is_warning_zone = False

        # Algorithm 1, line 12: only test when older window has >= w samples
        if self.no >= self.w:
            # Algorithm 1, lines 13-14: scale older window counts to size w
            wp = round(self.wo * self.w / self.no)
            wr = sum(self.recent)

            # Algorithm 4
            p_value = self._compute_p_value(wr, wp)
            self.stats.p_value = p_value
            self.stats.error_rate_reference = self.wo / self.no
            self.stats.error_rate_current = wr / self.w

            # Algorithm 1, lines 16-19
            if p_value < self.alpha_d:
                self.drift_detected = True
                self.stats.drift_count += 1
                return 'drift'
            elif p_value < self.alpha_w:
                self.is_warning_zone = True
                self.stats.warning_count += 1
                return 'warning'

        return 'normal'