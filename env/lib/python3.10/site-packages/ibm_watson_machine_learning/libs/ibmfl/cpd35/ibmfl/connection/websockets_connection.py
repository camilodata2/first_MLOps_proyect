"""Connection class which uses flask and request libraries to create a server
client combo
"""
#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
import socket

import ibm_watson_machine_learning._wrappers.requests as requests
import asyncio
import signal
import random
import websockets
import pathlib
from websockets import exceptions
import string
import argparse
import threading
import time
import ssl
from _collections import deque
import sys
import traceback

from ibmfl.connection.connection import ConnectionStatus
from ibmfl.connection.connection import FLConnection, FLSender, FLReceiver
from ibmfl.message.message_type import MessageType
from ibmfl.message.message import Message
from ibmfl.message.serializer_types import SerializerTypes
from ibmfl.message.serializer_factory import SerializerFactory

logger = logging.getLogger(__name__)


# Maintained only on the party side
# Used to signal to websockets that there is a message ready to be sent to the aggregator
pSendEvt = threading.Event()

# Used to signal that there is a message that was received from the aggregator
pRecvEvt = threading.Event()



# Used to store messages to be sent to aggregator
party_sbuffer = deque([])
# Used to store messages received from aggregator
party_rbuffer = deque([])

pRouter = None


class WSConnection(FLConnection):

    def __init__(self, config):
        """
        Initializes the connection object
        :param config: dictionary of configuration provided to connection
        :type config: `dict`
        """

        pSendEvt.clear()
        pRecvEvt.clear()
        # comes from the config file
        self.started = False
        self.stopped = False
        self.settings = config
        self.receiver = None
        self.sender = None
        self.loop1 = None
        self.loop2 = None
        self.stop1 = None
        self.stop2 = None
        self.mainLoop = None
        self.flagForOutstandingMessageResponseToBeReceived = False
        # identity/name of the party

        if 'id' in config:
            self.party = config['id']
        else:
            # TODO: Fix error if no rts id
            self.party = 'DEFAULT'

        # Send Port is used for messages initiated from aggregator
        if 'wssendport' in config:
            self.wssendport = config['wssendport']
        else:
            # set a default value to be aggregator
            self.wssendport = "/aggregator"
        # Recv port is used for messages initiated from party
        if 'wsrecvport' in config:
            self.wsrecvport = config['wsrecvport']
        else:
            # set a default value to be aggregator
            self.wsrecvport = "/aggregator1"


    async def PartySendLoop(self):
        """
        This is started at the party.
        It reads from the party_sbuffer, and synchronously sends each message to the aggregator
        and waits for a response from the aggregator.
        SendLoop is a misnomer, like in Flask Connection.
        It merely means that messages are initiated by the party.
        """

        uri = "wss://" + self.aggInfo['ip'] + str(self.wsrecvport)
        logger.debug("PartySendLoop started = " + uri)
        ssl_context_party = True
        if self.ssl_context is not None:
            ssl_context_party = ssl.SSLContext()

        while not self.stopped:
            if len(party_sbuffer) == 0:
                logger.info("PartySendLoop: Holding for message to send" )
                pSendEvt.wait()
            try:
                logger.info("PartySendLoop: Number of active messages ready to send: " + str(len(party_sbuffer)))
                msg = party_sbuffer.popleft()
            except IndexError:
                logger.debug("No message to send")
                continue
            pSendEvt.clear()
            logger.debug("PartySendLoop: Sending message to aggregator")
            headers = None
            if self.authToken != None:
                headers = { 'Authorization': self.authToken, 'rtsid': self.party }
            try:
                async with websockets.connect(uri, max_size=2 ** 29, read_limit=2 ** 29, write_limit=2**29, ssl=ssl_context_party, extra_headers=headers) as websocket:

                    if self.flagForOutstandingMessageResponseToBeReceived == False:
                        logger.debug(
                            "PartySendLoop: Attempting to send message to aggregator")
                        try:
                            await websocket.send(msg)

                        except Exception as ex:
                            logger.error("PartySendLoop exception when attempting to send message: " + str(ex))
                            party_sbuffer.appendleft(msg)
                            raise
                        logger.debug(
                            "PartySendLoop: Message sent successfully; awaiting message response")

                    self.flagForOutstandingMessageResponseToBeReceived = True
                    resp = await websocket.recv()
                    self.flagForOutstandingMessageResponseToBeReceived = False
                    party_rbuffer.append(resp)
                    pRecvEvt.set()

            except websockets.exceptions.ConnectionClosedError as ex:
                if self.stopped != True:
                    logger.error(
                        "PartySendLoop : Connection closed abnormally. Exception is " + str(ex))
                    logger.error("Retrying connection to aggregator")
                    continue
            except websockets.exceptions.ConnectionClosedOK as ex:
                if self.stopped != True:
                    logger.error(
                        "PartySendLoop : Connection closed unexpectedly. Exception is " + str(ex))
                    logger.error("Retrying connection to aggregator")
                    continue
            except websockets.exceptions.WebSocketException as ex:
                logger.error(
                    "PartySendLoop : WebsocketException. Details: " + str(ex))
                return False
            except socket.gaierror as ex:
                logger.error("PartySendLoop : Error connecting to " + uri + ". Exception is " + str(ex))
                return False
            except Exception as ex:
                logger.error("PartySendLoop : Exception is " + str(ex))
                return False
        logger.debug("PartySendLoop ended ")
        return True

    async def PartyRecvLoop(self):
        """
        This is started at the party.
        It establishes a persistent connection to the aggregator by sending the name of the party
        The aggregator can then send messages to the party synchronously
        to which the party replies
        """
        pName = self.party
        uri = "wss://" + self.aggInfo['ip'] + str(self.wssendport)
        logger.debug("PartyRecvLoop started = " + uri)

        ssl_context_party2 = True
        if self.ssl_context is not None:
            ssl_context_party2 = ssl.SSLContext()

        headers = None
        if self.authToken != None:
            headers = { 'Authorization': self.authToken }
        while not self.stopped:
            try:
                async with websockets.connect(uri, close_timeout=100, max_size=2 ** 29, read_limit=2 ** 29, write_limit=2**29, ssl=ssl_context_party2, extra_headers=headers) as websocket:
                    await websocket.send(pName)
                    async for rmess in websocket:
                        if rmess == "HEARTBEAT":
                            logger.info("Received Heartbeat from Aggregator")
                            continue
                        serializer = SerializerFactory(
                            SerializerTypes.JSON_PICKLE).build()
                        recv_msg = serializer.deserialize(rmess)
                        logger.debug(
                            "PartyRecvLoop : Received " + str(recv_msg))
                        request_path = str(recv_msg.message_type)
                        logger.debug(
                            "REQUEST PATH IN PARTY IS " + str(request_path))

                        # check if error
                        if recv_msg.message_type == MessageType.ERROR_AUTH.value :
                            logger.info(
                            "PartyRecvLoop : Auth Error. Please verify training ID, RTS, project permission and token")
                            self.stop()
                            break;

                        handler, kwargs = self.router.get_handler(
                            request_path=request_path)

                        if handler is None:
                            logger.error(
                                'PartyRecvLoop : Invalid Request ! Routing it to default handler')
                            handler, kwargs = self.router.get_handler(
                                request_path='default')
                        try:
                            res_message = handler(recv_msg)
                        except Exception as ex:
                            res_message = Message()
                            data = {'status': 'error', 'message': str(ex)}
                            res_message.set_data(data)

                        response = serializer.serialize(res_message)
                        await websocket.send(response)
                        logger.info("PartyRecvLoop : sent response")

                        if recv_msg.message_type == MessageType.STOP.value :
                            logger.info(
                            "PartyRecvLoop : Received STOP message" )
                            self.stop()
                            break;

            except websockets.exceptions.ConnectionClosedError as ex:
                if self.stopped != True:
                    logger.error(
                        "PartyRecvLoop : Connection closed abnormally. Exception is " + str(ex))
                    logger.error("Retrying connection to aggregator")
                    continue
            except websockets.exceptions.ConnectionClosedOK as ex:
                if self.stopped != True:
                    logger.error(
                        "PartyRecvLoop : Connection closed unexpectedly. Exception is " + str(ex))
                    logger.error("Retrying connection to aggregator")
                    continue
            except websockets.exceptions.WebSocketException as ex:
                logger.error(
                    "PartyRecvLoop : WebsocketException. Details: " + str(ex))
                return False
            except socket.gaierror as ex:
                logger.error("PartyRecvLoop : Error connecting to " + uri + ". Exception is " + str(ex))
                return False
            except Exception as ex:
                logger.error("PartyRecvLoop : Exception is " + str(ex))
                return False
        logger.debug("PartyRecvLoop ended ")
        return True

    def PartySendLoopThread(self, loop):
        logger.info("**** PartySendLoopThread")
        # Need to host the loop in its own thread for asyncio to work properly
        pslt_result = loop.run_until_complete(self.PartySendLoop())
        logger.debug("**** PartySendLoopThread RESULT = " + str(pslt_result))
        if pslt_result == False:
            #if error when sending message, then unblock mainThread so it can fail.
            pRecvEvt.set()
            self.stop()

    def PartyRecvLoopThread(self, loop):
        logger.info("**** PartyRecvLoopThread")
        # Need to host the loop in its own thread for asyncio to work properly
        prlt_result = loop.run_until_complete(self.PartyRecvLoop())
        logger.debug("**** PartyRecvLoopThread RESULT = " + str(prlt_result))
        if prlt_result == False:
            self.stop()


    def initialize(self, router, aggInfo=None, ssl_context=None, authToken=None):
        """Initialize receiver and sender """
        self.router = router
        self.ssl_context = ssl_context

        if aggInfo == None:
            logger.error("Aggregator information not specified. Exiting.")
            exit()
        logger.info("WSConnection : Initialize Party Communications")
        self.party = self.settings['id']
        self.aggInfo = aggInfo
        self.authToken = authToken

        # Each websocket connection needs a separate asyncio event loop
        self.loop1 = asyncio.new_event_loop()
        t1 = threading.Thread(
            target=self.PartySendLoopThread, args=(self.loop1,), daemon=True)
        t1.start()

        # Each websocket connection needs a separate asyncio event loop
        self.loop2 = asyncio.new_event_loop()
        t2 = threading.Thread(
            target=self.PartyRecvLoopThread, args=(self.loop2,), daemon=True)
        t2.start()

    def initialize_receiver(self, router=None):
        """Basically does nothing
        :param router: Router object describing the routes for each request
            which are passed down to PH
        :type router: `Router`
        """
        # print(router)
        self.router = router
        pRouter = router
        self.receiver = WSReceiver(router, self.settings, self)
        self.receiver.connection = self
        logger.debug('Receiver Initialized')
        self.receiver.initialize()
        self.status = ConnectionStatus.INITIALIZED

    def initialize_sender(self):
        """BAsically does nothing
        """
        self.sender = WSSender(self.settings)
        self.sender.initialize()
        self.status = ConnectionStatus.INITIALIZED

    def start(self):
        """
        Set self.started to True
        """
        # self.start_receiver()
        self.started = True
        self.status = ConnectionStatus.STARTED
        self.SENDER_STATUS = ConnectionStatus.STARTED

    def stop(self):
        """
        set self.stopped to true
        """
        self.stopped = True
        logger.debug('Stopping Receiver and Sender')
        pSendEvt.set()
        self.status = ConnectionStatus.STOPPED
        try:
            if self.stop1 is not None:
                self.stop1.cancel()
            if self.stop2 is not None:
                self.stop2.set_result(None)
            if self.mainLoop is not None:
                self.mainLoop.call_soon_threadsafe(self.mainLoop.stop)
        except Exception as ex:
            logger.error("Exception when stopping connection: " + str(ex))

    def get_connection_config(self):
        pass

class WSReceiver(FLReceiver):

    def __init__(self, router, settings, connection):
        """
        Does nothing
        """

        self.router = router
        self.settings = settings
        self.connection = connection

    def shutdown_server(self):
        pass

    def initialize(self):
        """
        Does nothing
        """
        logger.debug('Initializing WSReceiver')


    def receive_message(self, party):
        logger.error('OOOOO We are in trouble. This should not be called')

    def start(self):
        pass

    def stop(self):
        pass


class WSSender(FLSender):

    """
    Basic implementation using the request package of python to send
    requests to Rest endpoints.
    """

    def __init__(self, settings):
        """
        Does nothing
        """
        self.settings = settings
        logger.debug("WS Sender Init. My info is " + str(settings))

    def initialize(self):
        """Does all the setups required for the sender. This method should also
        check for availability of resources, open ports, certificates etc.,
        if required
        """
        logger.info('Websockets Sender initialized')

    def send_message(self, destination, message):
        """
        used for sending all the requests. Message object should be
        validated and endpoint should be decided based on message codes.
        :param destination: information about the destination to which message
        should be forwarded
        :type destination: `dict`
        :param message: message object constructed by aggregator/party
        :type message: `Message`
        :return : response object
        :rtype : `Response`
        """
        if 'id' in self.settings:
            message.add_sender_info(self.settings['id'])
        else:
            # set a default value to be Aggregator
            message.add_sender_info('Aggregator')

        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()
        message = serializer.serialize(message)

        logger.info("Sending serialized message to aggregator")
        party_sbuffer.append(message)
        pSendEvt.set()
        pRecvEvt.wait()
        response = None
        try:
            response = party_rbuffer.popleft()
        except IndexError:
            logger.debug("[WSSender] No message response received, after attempt to send message")

        pRecvEvt.clear()
        response_message = None
        if response is not None:
            # inspect response
            if isinstance(response, str):
                logger.info('Received String message as response: ' + response)
                response_message = Message(
                    MessageType.REGISTER.value, data={'status': 'success', 'response': response})
            else:
                # deserialize response
                response_message = serializer.deserialize(response)
                logger.info(
                    'Received serialized message as response: ' + str(response_message))

        return response_message

    def cleanup(self):
        """
        Cleanup the Sender object and close the hanging connections if any
        """
        logger.info('Cleaning up Websockets Client')
