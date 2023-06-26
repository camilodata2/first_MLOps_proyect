#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

class ArtifactReader(object):
    """
    Template reader class used to read artifacts.
    """

    def read(self):
        """
        Returns stream object with content of pipeline/pipeline model.

        :return: binary stream
        """
        pass

    def close(self):
        """
        Closes stream to content.
        """
        pass
