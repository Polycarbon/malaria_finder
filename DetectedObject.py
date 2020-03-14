from collections import defaultdict


class ObjectMap(defaultdict):
    def __missing__(self, key):
        return None
