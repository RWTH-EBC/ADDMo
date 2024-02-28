from extrapolation_detection.detector.detector_factory import DetectorFactory
from extrapolation_detection.use_cases.train_clf import train_clf_ideal
from extrapolation_detection.detector.config.detector_config import DetectorConfig

def exe_train_clf_ideal(config: DetectorConfig):
    for detector_name in config.detectors:
        train_clf_ideal(config., clf_name, clf_callback, outlier_fraction, beta=beta)