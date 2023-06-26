#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

class Party:

    def __init__(self, **kwargs):
        from ibm_watson_machine_learning.party_wrapper import Party
        self.party = Party(**kwargs)
        self.connection = self.party.connection

    def start(self):
        self.party.start()