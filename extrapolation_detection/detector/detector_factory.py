import inspect
from extrapolation_detection.detector import detector_tuners
from extrapolation_detection.detector import detectors
from extrapolation_detection.detector.abstract_detector import AbstractDetector
from extrapolation_detection.detector.detector_tuners import AbstractHyper

class DetectorFactory:
    """
    Creates and returns an instance of the specified detector.
    """
    @staticmethod
    def detector_factory(detector_type: str) -> AbstractDetector:
        """Get the detector instance dynamically."""

        detector_type = "D_" + detector_type

        # If detector is based on extrapolation_detection.detector
        if hasattr(detectors, detector_type):
            custom_detector_class: detectors.AbstractDetector = getattr(detectors, detector_type)
            return custom_detector_class()

        # If detector is not found
        else:
            # Get the names of all custom detectors for error message
            custom_detector_names = [
                name
                for name, obj in inspect.getmembers(detectors)
                if inspect.isclass(obj)
                and issubclass(obj, AbstractDetector)
                and not inspect.isabstract(obj)
            ]

            raise ValueError(
                f"Unknown detector type: {detector_type}. "
                f"Available custom detectors are: {', '.join(custom_detector_names)}. "
            )

    @staticmethod
    def detector_tuner_factory(detector_type: str) -> AbstractHyper:
        """Get the detector instance dynamically."""

        detector_tuner_type = "Hyper_" + detector_type

        # If detector is based on extrapolation_detection.detector
        if hasattr(detector_tuners, detector_tuner_type):
            custom_detector_class: detector_tuners.AbstractHyper = getattr(detector_tuners,
                                                                detector_tuner_type)
            return custom_detector_class()

        # If detector is not found
        else:
            # Get the names of all custom detectors for error message
            custom_detector_names = [
                name
                for name, obj in inspect.getmembers(detector_tuners)
                if inspect.isclass(obj)
                and issubclass(obj, AbstractDetector)
                and not inspect.isabstract(obj)
            ]

            raise ValueError(
                f"Unknown detector tuner type: {detector_tuner_type}. "
                f"Available custom detector tuner (Hyper_<DetectorName>) are:"
                f" {', '.join(custom_detector_names)}. "
            )