#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

class MetaProps:
    """
    Holder for props used during creation of ML artifacts.

    :param map meta: Map of pair key and value where key is taken from MetaNames.
    """
    def __init__(self, meta):
        self.meta = meta

    def available_props(self):
        """Return list of strings with names of available props."""
        return self.meta.keys()

    def prop(self, name):
        """Get prop value by name."""
        return self.meta.get(name)

    def merge(self, other):
        """Merge other MetaProp object to first one. Modify first MetaProp object, doesn't return anything."""
        self.meta.update(other.meta)

    def add(self, name, value):
        """
        Add new prop.

        :param str name: Key for value. Should be one of the values from MetaNames.
        :param object value: Any type of object
        """
        self.meta[name] = value

    def get(self):
        """returns meta prop dict"""
        return self.meta
