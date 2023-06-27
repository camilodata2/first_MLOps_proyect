#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
import asyncio
import threading
import websockets
from _collections import deque
import signal
import ssl
import time
import concurrent.futures._base

from ibmfl.connection.connection import ConnectionStatus
from ibmfl.connection.connection import FLConnection, FLSender, FLReceiver
from ibmfl.message.message import Message, ResponseMessage
from ibmfl.message.message_type import MessageType
from ibmfl.message.serializer_types import SerializerTypes
from ibmfl.message.serializer_factory import SerializerFactory

logger = logging.getLogger(__name__)

#TODO: remove  self.temp_rts_id = '4a666f4a-87c0-475d-bfb2-312d186163b4' 

party_rbuffer = deque([])
party_sbuffer = deque([])
pRecvEvt = threading.Event()
pRecvEvt.clear()
pSendEvt = threading.Event()
pSendEvt.clear()

class SWSConnection(FLConnection):

    DEFAULT_HOST = '127.0.0.1'
    DEFAULT_PORT = 8765
    TEMP_rts_id = '4a666f4a-87c0-475d-bfb2-312d186163b4' 
    DEFAULT_CONFIG = {'ip': DEFAULT_HOST, 'port': DEFAULT_PORT, 'rts_id': TEMP_rts_id}
    rts_id = None



    def __init__(self, config):
        self.stopped = False   # flag has been exposed in generated script.
        self.settings = self.process_config(config)
        self.rts_id = self.settings.get('id') 
        logger.debug("METHOD = SWSConnection.init")
        logger.debug(self.settings)


    def initialize(self, **kwargs):
        pass

    def initialize_receiver(self, **kwargs):
        self.router = kwargs.get('router')
        self.settings['verify'] = kwargs.get('verify') if kwargs.get('verify') is not None else True
        self.settings['token'] = kwargs.get('token')
        #TODO: Check if sender is intialized, otherwise ?
        self.receiver = SWSReceiver(self.router, self.sender, self.rts_id, self.settings)
        self.receiver.initialize()
        self.receiver.stoppedStateObj = self

    def initialize_sender(self, **kwargs):
        
        self.sender = SWSSender(self.rts_id, self.settings)
        self.sender.initialize()

    def start(self, **kwargs):
        self.receiver.start()


    def stop(self, **kwargs):
        if self.receiver.stop is not None:
            self.receiver.stop.set_result
            self.stopped = True

    def get_connection_config(self):

        return self.settings


    def process_config(self, config): 
        logger.debug("METHOD = SWSConnection.process_config")
        logger.debug(config)

        if config.get('ip', None) is not None and config.get('port', None) is not None:
            return config
        else:
            return self.DEFAULT_CONFIG


class SWSReceiver(FLReceiver):

    def __init__(self, router, sender, party_id, settings):
        self.router = router
        self.ssl_context = True
        self.sender = sender
        self.stopped = False
        self.stopping = False
        self.party = party_id
        self.settings = settings
        self.stop = None
        self.messageInTransit = None
        self.stoppedStateObj = None

    def initialize(self, **kwargs):
        pass

    def debug_websocket(self, websocket):
        #pass
        logger.debug("*************")
        logger.debug(websocket.remote_address)
        logger.debug(websocket.request_headers.get("rtsid"))
        logger.debug("*************")

    def prepare_message_data_structures(self):
            # party_rbuffer = deque([])
            # party_sbuffer = deque([])
            # pRecvEvt = threading.Event()
            # pRecvEvt.clear()
            # pSendEvt = threading.Event()
            # pSendEvt.clear()
        pass
    

    def start(self, **kwargs):
        logger.debug("METHOD = SWSReceiver.start")
        self.loop = asyncio.new_event_loop()
        self.stop = self.loop.create_future()
        self.loop.add_signal_handler(signal.SIGTERM, self.stop.set_result, None)
        t1 = threading.Thread(target=self.setup_websocket_connection, args=(self.loop, self.stop), daemon=True)
        t1.start()
       

    def stop(self, **kwargs):
        logger.debug("METHOD = SWSReceiver.stop")
        self.stopped = True
        if self.stoppedStateObj is not None:
            self.stoppedStateObj.stopped = True
        

    def setup_websocket_connection(self, loop, stop):
        logger.debug("METHOD = SWSReceiver.setup_websocket_connection")
        try:
            loop.run_until_complete(self.establish_websocket_connection(loop, stop))

            #asyncio.get_event_loop().run_forever()
        except websockets.exceptions.ConnectionClosedError as ex:
            if self.stopped != True:
                logger.error(
                    "PartySendLoop : Connection closed abnormally. Exception is " + str(ex))
                logger.error("Retrying connection to aggregator")
        except websockets.exceptions.ConnectionClosedOK as ex:
            if self.stopped != True:
                logger.error(
                    "PartySendLoop : Connection closed unexpectedly. Exception is " + str(ex))
                logger.error("Retrying connection to aggregator")
            else:
                logger.debug("Connection Closed OK")
        except websockets.exceptions.WebSocketException as ex:
            logger.error("PartySendLoop : WebsocketException. Details: " + str(ex))
        except Exception as ex:
            logger.error("PartySendLoop : Exception is " + str(ex))
        
        self.stop.set_result


    async def establish_websocket_connection(self, loop, stop):
        logger.debug("METHOD = SWSReceiver.establish_websocket_connection")
        uri = "wss://" + self.settings.get("ip") + ":" + str(self.settings.get("port"))
        logger.debug("SWSReceiver.establish_websocket_connection uri = "+uri)

        headers = { 'rtsid': self.party }
        token = self.settings.get('token')
        if ( token is not None ):
            headers['Authorization'] = token
        if ( not self.settings.get('verify') ) :
            self.ssl_context = ssl.SSLContext()
        logger.debug("SWSReceiver.establish_websocket_connection ssl = "+str(self.ssl_context))
        apikey_header = None
        while not self.stopped:
            try:
                logger.debug("SWSReceiver.establish_websocket_connection ATTEMPT async websockets.connect")
                async with websockets.connect(uri, close_timeout=1000, max_size=2 ** 29, read_limit=2 ** 29, write_limit=2**29, ssl=self.ssl_context, loop=loop, extra_headers=headers) as websocket:
                    logger.info("Received Heartbeat from Aggregator")
                    apikey_header = websocket.response_headers.get("apikey", None)
                    if apikey_header is not None:
                        logger.debug("API KEY received from Aggregator, for reconnection authorization")
                        headers["Authorization"] = apikey_header
                    await self.websocket_message_loop(websocket)
            except websockets.exceptions.ConnectionClosedError as ex:
                logger.info("ConnectionClosedError encountered.")
                logger.error(str(ex))

                if self.stopped != True:
                    logger.error(
                        "PartySendLoop : Connection closed abnormally. Exception is " + str(ex))
                    logger.info("Retrying connection to aggregator")
                    continue
            except websockets.exceptions.ConnectionClosedOK as ex:
                logger.info("ConnectionClosedOK encountered.")
                logger.error(str(ex))
                if self.stopped != True:
                    logger.error(
                        "PartySendLoop : Connection closed unexpectedly. Exception is " + str(ex))
                    logger.info("Retrying connection to aggregator")
                    continue
                else:
                    logger.info("Connection Closed OK")
                    break
            except websockets.exceptions.WebSocketException as ex:
                logger.info("WebsocketException encountered.")
                logger.error(str(ex))
                self.stopped = True
                continue
            except Exception as ex:
                logger.info("Exception encountered.")
                logger.error(str(ex))
                break

    async def websocket_message_loop(self, websocket):
        logger.debug("METHOD = SWSReceiver.websocket_message_loop")
        self.debug_websocket(websocket)

        receive_message_task = asyncio.ensure_future(self.receive_message_handler(websocket) )
        send_message_task = asyncio.ensure_future(self.send_message_handler(websocket) )
        stop_loop_tasks = False
        while not stop_loop_tasks:
            try:

                done, pending = await asyncio.wait([receive_message_task, send_message_task], return_when=asyncio.FIRST_COMPLETED)

                logger.debug("*******************")
                logger.debug(done)
                logger.debug(pending)
                logger.debug("*******************")

                for task in pending:
                    task.cancel()
                for future in done:
                    if not future.cancelled():
                        logger.debug("Future Processed successfully")
                if len(pending) == 0 :
                    stop_loop_tasks = True
            except websockets.exceptions.ConnectionClosedError as ex:
                logger.exception("$$ EXCEPTION IN websocket_message_loop - ConnectionClosedError encountered.")
                self.invalidate_message_data_structures(websocket.request_headers.get("rtsid"), True)
                self.stopped = True

            except Exception as ex:
                logger.exception("$$ EXCEPTION IN WEBSOCKET MESSAGE LOOP for " + websocket.request_headers.get("rtsid") +"; Retrying connection")
                print(ex)

    async def receive_message_handler(self, websocket):
        logger.debug("METHOD = SWSReceiver.receive_message_handler")
        try:
            async for message in websocket:
                logger.debug("METHOD = SWSReceiver.receive_message_handler")
                self.debug_websocket(websocket)
                await self.process_received_message(message)
                #logger.info(message)
        except concurrent.futures._base.CancelledError as ex:
            logger.debug("task is being cancelled")
        except websockets.exceptions.ConnectionClosedError as ex:
            logger.info("ConnectionClosedError encountered.")
            if self.messageInTransit is not None:
                logger.info("Adding in Transit message back to queue")
                party_sbuffer.append(self.messageInTransit)
                self.messageInTransit = None
        except Exception as ex:
            print(ex)

    async def send_message_handler(self, websocket):
        while self.stopped is not True:
            logger.debug("METHOD = SWSReceiver.send_message_handler")
            self.debug_websocket(websocket)
            message = await self.process_message_to_send(websocket.request_headers.get("rtsid"))
            try:
                if message is not None:
                    await websocket.send(message)
                    self.messageInTransit = message
                    logger.debug("METHOD = SWSReceiver.send_message_handler Message SENT")
                    if self.stopping == True:
                        self.stop.set_result(True)
                        self.stopped = True
                        if self.stoppedStateObj is not None:
                            self.stoppedStateObj.stopped = True
            except websockets.exceptions.ConnectionClosedError as ex:
                logger.info("ConnectionClosedError encountered")
            except websockets.exceptions.ConnectionClosedOK as ex:
                logger.info("ConnectionClosedOK")
            except websockets.exceptions.WebSocketException as ex:
                logger.info("WebSocketException encountered")
            except Exception as ex:
                logger.info("Exception")
                logger.info(str(ex))

    async def process_message_to_send(self, rts_id):
        while self.stopped is not True:
            # if len(party_sbuffer) == 0:
            #     logger.info("pSendEvt.wait(), pSendEvt.is_set = " + str(pSendEvt.is_set()))
            #     pSendEvt.wait()
            try: 
                msgToBeSent = party_sbuffer.popleft()
                log_message = f"METHOD = SWSReceiver.process_message_to_send {msgToBeSent}"
                log_message = (log_message[:300]+ '...') if len(log_message) > 300 else log_message
                logger.debug(log_message)
                #logger.debug("METHOD = SWSReceiver.process_message_to_send ")

                return msgToBeSent
            except IndexError:
                await asyncio.sleep(1)
                continue
            except Exception as ex:
                logger.exception("$$ EXCEPTION IN process_message_to_send")
                print(ex)
    
    async def process_received_message(self, message):
        log_message = f"METHOD = SWSReceiver.process_received_message {message}"
        log_message = (log_message[:300]+ '...') if len(log_message) > 300 else log_message
        logger.debug(log_message)
        self.messageInTransit = None
        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()
        recv_msg = serializer.deserialize(message)
        message_type = str(recv_msg.message_type)
        logger.debug('METHOD = SWSReceiver.process_received_message type = ' + message_type )

        if (recv_msg.message_type == MessageType.STOP.value):
            handler, kwargs = self.router.get_handler(request_path=message_type)
            if handler is not None:
                res_message = handler(recv_msg)
                
            self.sender.send_message(recv_msg.sender_info, res_message, False)
            self.stopping = True
            return

        # check if error
        if (recv_msg.message_type == MessageType.ERROR_AUTH.value):
            logger.info("Authorization Error. Please verify ip address, training ID, RTS, project permission and token")
            self.stopped = True
            party_rbuffer.append(recv_msg)
            pRecvEvt.set()

            return

        # if recv_msg.sender_info is not None and recv_msg.sender_info == self.party:
        #     logger.info("Received Message is reply to previously sent message")
        #     return

        # msg_data = recv_msg.get_data()
        # if msg_data is not None and msg_data.get('status', None) is not None:
        #     res_message = self.handle_response_message()
        #     logger.info("status = " + msg_data.get('status'))
        #     self.sender.send_message(recv_msg.sender_info, res_message, False)
       


        handler, kwargs = self.router.get_handler(request_path=message_type)

        if handler is None:

            party_rbuffer.append(recv_msg)
            pRecvEvt.set()

        else:

            if recv_msg.sender_info is not None and recv_msg.sender_info == self.party:
                logger.debug("Received Message is reply to previously sent message")
                return
                
            try:
                res_message = handler(recv_msg)
            except Exception as ex:
                res_message = Message()
                data = {'status': 'error', 'message': str(ex)}
                res_message.set_data(data)


            self.sender.send_message(recv_msg.sender_info, res_message, False)

    def handle_stop_message(self):
        logger.info("Received STOP request from aggregator")
        response_msg = ResponseMessage(message_type=MessageType.STOP.value,
                                        id_request=-1,
                                        data={'ACK': True})
        logger.info("received a STOP request")
        time.sleep(30)
        self.stopped = True
        return response_msg

    def handle_response_message(self):
        logger.debug("Received response from aggregator with status payload")
        response_msg = ResponseMessage(message_type=MessageType.ACK.value,
                                            id_request=88888,
                                            data={'ACK': True})
        return response_msg    


class SWSSender(FLSender):

    def __init__(self, source_info, settings):

        self.settings = settings
        self.source_info = source_info

    def initialize(self, **kwargs):
        pass

    def send_message(self, destination, message, expectResponse=False):
        message.add_sender_info(self.source_info)
        message_type = str(message.message_type)
        logger.debug('SWSSender.send_message message_type = ' + message_type + ' to destination = ' + str(destination))
        if (message.message_type == MessageType.REGISTER.value):
            expectResponse = True
            logger.debug('SWSSender.send_message set expectResponse to True for REGISTER message')
        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()
        message = serializer.serialize(message)
        party_sbuffer.append(message)
        # pSendEvt.set()
        logger.debug("############# Thread switch")
        if expectResponse:
            logger.debug("############# Waiting for response")
            pRecvEvt.wait()
            logger.debug("############# pRecvEvt activated")
            response_message = party_rbuffer.popleft()
            pRecvEvt.clear()
            logger.debug(
                'Received serialized message as response: ' + str(response_message))

            return response_message

    
    def cleanup(self):
        pass
