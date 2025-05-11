# concept_drift/ddm.py
from river.drift import KSWIN

class KSWINDriftDetector:
    def __init__(self):
        self.detector = KSWIN()
        self.drift_detected = False

    def update(self, prediction, true_label):
        # Update the drift detector with the error
        self.detector.update(prediction != true_label)
        if self.detector.drift_detected:
            self.drift_detected = True
            # self.detector.reset()  # Reset the detector after drift is detected
        else:
            self.drift_detected = False
