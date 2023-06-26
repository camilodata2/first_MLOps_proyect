#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import psutil
import sys
import time
import threading
from builtins import property

class PSEntry(object):
    def __init__(self, t=None, io_counters=None, memory_info=None):
        self.time = t
        self.io_counters = io_counters
        self.memory_info = memory_info


class PSProfiler(object):
    '''A help class to profile process using `psutil`
    '''
    def __init__(self):
        self.psthread = None
        self.psstop = False
        self.psstorage = []
        self.proc = psutil.Process()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        def instr_io(proc, storage):
            while(not self.psstop):
                entry = PSEntry(t=time.time(),
                                io_counters=proc.io_counters(),
                                memory_info=proc.memory_info())
                storage.append(entry)
                time.sleep(1)

        self.psthread = threading.Thread(target=instr_io,
                                         args=(self.proc,
                                               self.psstorage))
        self.psthread.start()

    def stop(self):
        self.psstop = True
        self.psthread.join()

    @property
    def mbyte_count(self):
        startpio = self.psstorage[0].io_counters
        endpio = self.psstorage[-1].io_counters
        bc = (endpio.other_bytes - startpio.other_bytes) / (1024*1024)
        return bc

    @property
    def elapsed_time(self):
        t0 = self.psstorage[0].time
        t_last = self.psstorage[-1].time
        return t_last - t0

    @property
    def rate(self):
        return self.mbyte_count / self.elapsed_time

    def write_info(self, filelike):
        filelike.write(f"Number of bytes read: {self.mbyte_count} MB, rate={self.rate} MB/sec, elapsed={self.elapsed_time}\n")

    def print_info(self):
        self.write_info(sys.stdout)

    def save_csv(self, outputname):
        '''Writes a csv with columns `time,io_count,io_bytes,rss,vms`
        '''
        t0 = self.psstorage[0].time
        startpio = self.psstorage[0].io_counters;
        startmem = self.psstorage[0].memory_info;
        with open(outputname, mode="w") as s:
            s.write("time,io_count,io_bytes,rss,vms")
            for e in self.psstorage:
                t = e.time
                c = e.io_counters
                m = e.memory_info
                elapsed = t-t0
                io_count = c.other_count - startpio.other_count
                io_bytes = c.other_bytes - startpio.other_bytes
                rss = m.rss - startmem.rss
                vms = m.vms - startmem.vms
                s.write(f"{elapsed},{io_count},{io_bytes},{rss},{vms}\n")



class IOProfiler(object):
    '''A help class to profile process I/O using `psutil`
    '''
    def __init__(self):
        self.psthread = None
        self.psstop = False
        self.psstorage = []
        self.proc = psutil.Process()

    def start(self):
        def instr_io(proc, storage):
            while(not self.psstop):
                storage.append((time.time(), proc.io_counters()))
                time.sleep(1)

        self.psthread = threading.Thread(target=instr_io,
                                         args=(self.proc,
                                               self.psstorage))
        self.psthread.start()

    def stop(self):
        self.psstop = True
        self.psthread.join()

    @property
    def mbyte_count(self):
        t0, startpio = self.psstorage[0]
        t_last, endpio = self.psstorage[-1]
        bc = (endpio.other_bytes - startpio.other_bytes) / (1024*1024)
        return bc

    @property
    def elapsed_time(self):
        t0, startpio = self.psstorage[0]
        t_last, endpio = self.psstorage[-1]
        return t_last - t0

    @property
    def rate(self):
        return self.mbyte_count / self.elapsed_time

    def write_info(self, filelike):
        filelike.write(f"Number of bytes read: {self.mbyte_count} MB, rate={self.rate} MB/sec, elapsed={self.elapsed_time}\n")

    def print_info(self):
        self.write_info(sys.stdout)

    def save_csv(self, outputname):
        '''Writes a csv with columns `time,io_count,io_bytes`
        '''
        t0, startpio = self.psstorage[0]

        with open(outputname, mode="w") as s:
            s.write("time,io_count,io_bytes")
            for t, c in self.psstorage:
                elapsed = t-t0
                io_count = c.other_count - startpio.other_count
                io_bytes = c.other_bytes - startpio.other_bytes
                s.write(f"{elapsed},{io_count},{io_bytes}\n")
