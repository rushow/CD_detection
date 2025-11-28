# from river.drift import ADWIN

# class ADWINDriftDetector:
#     def __init__(self):
#         self.detector = ADWIN(min_window_length = 100, delta=0.002)
#         self.drift_detected = False

#     def update(self, prediction, true_label):
#         # Update the drift detector with the error
#         self.detector.update(prediction != true_label)
#         if self.detector.drift_detected:
#             self.drift_detected = True
#             # self.detector.reset()  # Reset the detector after drift is detected
#         else:
#             self.drift_detected = False



from collections import deque
from statistics import mean
from river.drift import ADWIN

class ADWINDriftDetector:
    def __init__(self, W=20, delta=0.002):
        self.W = W
        self.delta = delta
        self.detector = ADWIN(delta=self.delta)
        self.buf = deque(maxlen=W)
        self.drift_detected = False

    def reset(self):
        self.detector = ADWIN(delta=self.delta)
        self.buf.clear()
        self.drift_detected = False

    def update(self, prediction, true_label):
        err = int(prediction != true_label)
        self.buf.append(err)

        # only feed when buffer is "full" to emulate width W
        if len(self.buf) == self.buf.maxlen:
            # aggregated error (e.g., mean over W)
            agg = mean(self.buf)
            self.detector.update(agg)

            self.drift_detected = bool(
                getattr(self.detector, "change_detected", False) or
                getattr(self.detector, "drift_detected", False)
            )
        else:
            self.drift_detected = False
