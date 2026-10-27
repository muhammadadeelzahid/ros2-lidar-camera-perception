"""Object classes the perception stack cares about.

In simulation a model counts as a hazard when its name starts with one of the
prefixes below. Gazebo reports poses but not sizes, so each class has a nominal box.
"""

# class name -> nominal box size (dx, dy, dz) in metres
CLASS_SIZE = {
    'Pedestrian': (0.6, 0.6, 1.8),
    'Vehicle': (1.2, 0.8, 1.4),
}

# model-name prefix -> class name
NAME_PREFIX_TO_CLASS = {
    'pedestrian': 'Pedestrian',
    'person': 'Pedestrian',
    'forklift': 'Vehicle',
    'vehicle': 'Vehicle',
}

CLASS_NAMES = list(CLASS_SIZE.keys())


def classify(model_name):
    """Return the class name for a model, or None if it is not a hazard."""
    lowered = model_name.lower()
    for prefix, cls in NAME_PREFIX_TO_CLASS.items():
        if lowered.startswith(prefix):
            return cls
    return None
