from .one_euro import OneEuroFilter

class PoseSmoother:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.filters = {}  # (joint, axis) -> OneEuroFilter

    def _get(self, joint, axis):
        key = (joint, axis)
        if key not in self.filters:
            self.filters[key] = OneEuroFilter(freq=self.freq, min_cutoff=self.min_cutoff, beta=self.beta)
        return self.filters[key]

    def smooth(self, keypoints):
        # keypoints: name -> (x,y,z,v)
        smoothed = {}
        for name, (x,y,z,v) in keypoints.items():
            x_s = self._get(name,'x').apply(x)
            y_s = self._get(name,'y').apply(y)
            z_s = self._get(name,'z').apply(z)
            smoothed[name] = (x_s, y_s, z_s, v)
        return smoothed
