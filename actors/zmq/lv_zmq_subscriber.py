from collections import deque # for deque circular/ring buffer, remove rear element, add front element
import numpy as np
import os
import pandas as pd
import time
# import traceback

import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, ZMQError, ETERM, EAGAIN
from zmq.log.handlers import PUBHandler

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, filename='lv_zmq_subscriber.log', encoding='utf-8')

logger.setLevel(logging.INFO)

class ZMQAcquirer(Actor):
    """
    TODO CONSIDER POLLING
    https://github.com/zeromq/pyzmq/blob/main/examples/monitoring/zmq_monitor_class.py
    """

    def __init__(self, *args, ip=None, port=None, multipart=True, fs=None, win_dur=None, time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        # single, global Context instance
        # for classes that depend on Contexts, use a default argument to enable programs with multiple Contexts, not require argument for simpler applications
        # called in a subprocess after forking, a new global instance is created instead of inheriting a Context that won’t work from the parent process
        # add self.ctx for global, different methods = PUB/SUB
        # self.name = "ZMQAcquirer"

        self.context = zmq.Context.instance()

        self.ip = ip
        self.port = port

        self.multipart = multipart

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path
            
            os.makedirs(self.out_path, exist_ok=True)
            logger.info(f"Timing directory for {self.name}: {self.out_path}")
        
        # n per seg, samples for given window duration, for example, 20 ms of data at 32000 fs = 20 ms of data at 320 fms
        # or 0.02 s * 32000 fs
        self.n_per_seg = win_dur * fs


    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"


    def setup(self):
        """
        TODO HERE SYNC SUB w/PUB w/REQ/REP
        https://zguide.zeromq.org/docs/chapter2/#Node-Coordination
        """

        logger.info(f"Running setup for {self.name}")

        logger.info("Setting up subscriber receive socket for acquisition")
        self.setRecvSocket() 
        
        # initialize timing lists
        if self.time_opt is True:
            logger.info(f"Initializing lists for {self.name} timing")
            
            self.zmq_acq_timestamps = []
            self.recv_msg = []
            self.get_data = []
            self.put_seg_to_store = []
            self.put_out_time = []
            self.zmq_acq_total_times = []


        self.dropped_msg = []
        self.ns_per_msg = []
        self.data = []
        # self.ns = deque([])
        # self.data = deque([], maxlen=int(1000))

        self.msg_num = 0
        self.seg_num = 0

        self.done = False

        logger.info(f"Completed setup for {self.name}")


    def stop(self):
        """
        Meh... https://stackoverflow.com/questions/9019873/should-i-close-zeromq-socket-explicitly-in-python
        """
        
        logger.info(f"{self.name} stopping")
        # Close subscriber socket
        logger.info("Closing subscriber socket")
        self.recv_socket.close()
        # Terminate context = ONLY terminate if/when BOTH SUB and PUB sockets are closed
        logger.info("Terminating context")
        self.context.term()

        if self.time_opt is True:
            logger.info(f"Saving out timing info for {self.name}")
            keys = self.timing
            values = [self.zmq_acq_timestamps, self.get_data, self.put_seg_to_store, self.put_out_time, self.zmq_acq_total_times]

            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.out_path, 'zmq_acq_timing.csv'), index=False, header=True)

        np.save(os.path.join(self.out_path, 'ns_per_msg.npy'), self.ns_per_msg)
        
        logger.info(f"{self.name} stopped")

        return 0
    
    
    def runStep(self):
        """
        TIMING — DOES IT TAKE LONGER TO APPEND LIST OR SET NEW VAR???
        REFACTOR?
        """

        if self.done:
            pass

        t = time.time()

        if self.time_opt is True:
            self.zmq_acq_timestamps.append(time.time())

        try:   
            # logger.info(f"Receiving msg {self.msg_num}")

            t1 = time.time()
            msg = self.recvMsg()
            t2 = time.time()
            # self.recv_msg.append((time.time() - t1)*1000.0)

            self.data.extend(np.int16(msg[:-1]))
            self.ns_per_msg.append(int(msg[-1]))

            # ns_obj_id = self.client.put(int(msg[-1]), f"n_per_msg_{self.msg_num}")

            # logger.info(f"data: {self.data}")
            # logger.info(f"ns: {self.ns}")

            # logger.info(f"seg {self.seg_num}, n={int(msg[-1])}")

            t3 = time.time()
            # self.get_data.append((time.time() - t2)*1000.0)

            if np.sum(self.ns_per_msg) >= self.n_per_seg * (self.seg_num + 1):

                data_obj_id = self.client.put(np.array(self.data[0:self.n_per_seg]), f"seg_num_{str(self.seg_num)}")
                seg_obj_id = self.client.put(self.seg_num, f"seg_num_{str(self.seg_num)}")
                self.put_seg_to_store((time.time() - t3)*1000.0)

                t4 = time.time()
                self.q_out.put(data_obj_id)
                self.put_out_time.append((time.time() - t4)*1000.0)
                
                self.data = self.data[self.n_per_seg:]
                self.seg_num += 1

            self.recv_msg.append((t2 - t1)*1000.0)
            self.get_data.append((t3- t2)*1000.0)

            self.msg_num += 1

        except Exception as e:
                logger.error(f"Acquirer general exception: {e}")
        except IndexError as e:
            pass

        # # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
        # except ObjectNotFoundError:
        #     logger.error('Acquirer: Message {} unavailable from store, dropping'.format(self.seg_num))
        #     self.dropped_msg.append(self.seg_num)
        #     # self.q_out.put([1])
        # except KeyError as e:
        #     logger.error('Processor: Key error... {0}'.format(e))
        #     # Proceed at all costs
        #     self.dropped_wav.append(self.seg_num)
        # except Exception as e:
        #     logger.error('Processor error: {}: {} during segment number {}'.format(type(e).__name__,
        #                                                                                 e, self.seg_num))
        #     print(traceback.format_exc())
        #     self.dropped_wav.append(self.seg_num)
        self.zmq_acq_total_times.append((time.time() - t)*1000.0)

        logger.info(f"{self.name} avg time per run: {np.mean(self.zmq_acq_total_times)} ms")

        # logger.info(f"Acquire broke, avg time per segment: {np.mean(self.zmq_acq_total_times)} ms")
        # logger.info(f"Acquire got through {self.seg_num} segments")


    def setRecvSocket(self):
    # def setRecvSocket(self, ip, port):
        """
        ADAPTED CHANG PR ZMQPSActor — limit redundant work

        DO NOT OPEN NEW ZMQ CONTEXT:
        https://stackoverflow.com/questions/45154956/zmq-context-should-i-create-another-context-in-a-new-thread
        https://github.com/zeromq/pyzmq/issues/1172
        https://stackoverflow.com/questions/71312735/inter-process-communication-between-async-and-sync-tasks-using-pyzmq

        TODO: error handling: EINVAL, socket type invalid; EFAULT, context invalid; EMFILE, limit num of open sockets; ETERM, context terminated        

        Sets up the receive socket for the actor — subscriber.
        """

        # self.subscriber = self.context.socket(SUB)
        self.recv_socket = self.context.socket(SUB)

        # connect client node, socket,  with unkown or arbitrary network address(es) to endpoint with well-known network address
        # connect socket to peer address
        # endpoint = peer address:TCP port, source_endpoint:'endpoint'
        # IPv4/IPv6 assigned to interface OR DNS name:TCP port
        recv_address = f"tcp://{self.ip}:{self.port}"
        self.recv_socket.connect(recv_address)
        # receivig messages on all topics
        self.recv_socket.setsockopt(SUBSCRIBE, b"")

        # https://github.com/zeromq/pyzmq/blob/main/examples/pubsub/topics_sub.py
        # for example, timestamps, data
        # b'time: ' or b't' and b'data: ' or b'd'
        # self.recv_socket.setsockopt(SUBSCRIBE, b"t")
        # self.recv_socket.setsockopt(SUBSCRIBE, b"d")

        logger.info(f"Subscriber socket: {recv_address}")
        # why include a timeout? time how long it takes to connect?
        # time.sleep(timeout)


    def recvMsg(self):
        """
        ADAPTED CHANG PR ZMQPSActor — limit redundant work
        Receives a message from the controller — subscriber.

        DO NOT ADD NOBLOCK HERE
        With flags=NOBLOCK, this raises ZMQError if no messages have arrived
        Will give error: "Resource temporarily unavailable."
        With flags=NOBLOCK, this raises :class:`ZMQError` if no messages have
        arrived; otherwise, this waits until a message arrives.
        See :class:`Poller` for more general non-blocking I/O.
        https://github.com/zeromq/pyzmq/issues/1320
        https://github.com/zeromq/pyzmq/issues/36
        If server given time to receive message, then client when receiving-non-blocking will get message. (BUT WE DO NOT WANT TO WAIT!)
        """

        try:
            if self.multipart:
                msg = self.recv_socket.recv_multipart()
            else: 
                msg = self.recv_socket.recv()
        except ZMQError as e:
            logger.info(f"ZMQ error: {e}, {self.msg_num}")
            if e.errno == ETERM:
                pass           # Interrupted - or break if in loop
            if e.errno == EAGAIN:
                pass  # no message was ready (yet!)
            else:
                raise # real error
                # traceback.print_exc()
        
        # logger.info(f"Msg {self.msg_num} received: {msg}")
        return msg # process message
    

# FOR TESTING PURPOSES OUTSIDE OF IMPROV
if __name__ == "__main__":

    out_path = '/home/eao21/project-improv/improv/demos/ava/zmq_ex_ns/ns_per_msg.npy'

    sub_ip = "10.122.168.184"
    sub_port = "5555"

    time_opt = False

    zmq_acq = ZMQAcquirer(name="ZMQAcq", ip=sub_ip, port=sub_port, time_opt=time_opt)

    zmq_acq.setup()

    data = []
    ns_per_msg = []
    
    t = time.time()
    while 1:
        msg = zmq_acq.recvMsg()
        
        data.extend(np.int16(msg[:-1]))
        ns_per_msg.append(int(msg[-1]))

        np.save(os.path.join(out_path, 'ns_per_msg.npy'), ns_per_msg)
        # one = zmq_acq.runZMQAcquirer()
        # print(one - t - 1)