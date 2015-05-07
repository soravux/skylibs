import os
from os import listdir
from os.path import abspath, isdir, join
import fnmatch

from envmap import EnvironmentMap


class SkyDB:
    def __init__(self, path):
        """Creates a SkyDB.
        The path should contain folders named by YYYYMMDD (ie. 20130619 for June 19th 2013).
        These folders should contain folders named by HHMMSS (ie. 102639 for 10h26 39s).
        Inside these folders should a file named envmap.exr be located.
        """
        p = abspath(path)
        self.intervals_dates = [join(p, f) for f in listdir(p) if isdir(join(p, f))]
        self.intervals = list(map(SkyInterval, self.intervals_dates))


class SkyInterval:
    def __init__(self, path):
        """Represent an interval, usually a day.
        The path should contain folders named by HHMMSS (ie. 102639 for 10h26 39s).
        """
        matches = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, 'envmap.exr'):
                matches.append(join(root, filename))

        self.probes = list(map(SkyProbe, matches))
        if len(self.probes) > 0:
            self.sun_visibility = sum(1 for x in self.probes if x.sun_visible) / len(self.probes)
        else:
            self.sun_visibility = 0

    @property
    def date(self):
        return os.path.normpath(self.path).split(os.sep)[-1]


class SkyProbe:
    def __init__(self, path, format_='angular'):
        """Represent an environment map among an interval."""
        self.path = path
        self.envmap = EnvironmentMap(path, format_)

    @property
    def sun_visible(self):
        return self.envmap.data.max() > 5000

    @property
    def mean_light_vector(self):
        raise NotImplementedError()

    @property
    def time(self):
        return os.path.normpath(self.path).split(os.sep)[-2]

