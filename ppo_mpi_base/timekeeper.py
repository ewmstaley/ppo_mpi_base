'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import time
import datetime
import numpy as np

class TimeKeeper(object):

    def __init__(self, goal, window=10):
        self.start_time = time.time()
        self.buffer = []
        self.window = window
        self.total_frames = 0
        self.goal_frames = goal
        self.last_time = time.time()

    def log(self, new_total_frames):
        delta_frames = new_total_frames - self.total_frames
        new_time = time.time()
        delta_time = new_time - self.last_time

        self.total_frames = new_total_frames
        self.last_time = new_time

        self.buffer.append({
                "frames":delta_frames,
                "elapsed_time":delta_time,
                "fps":delta_frames/delta_time
            })

        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window:]

    def seconds_to_string(self, s):
        return str(datetime.timedelta(seconds=s))

    def thou(self, x):
        return f'{x:,}'

    def report(self):
        uptime = time.time() - self.start_time
        fps = np.mean([b["fps"] for b in self.buffer])
        remaining = self.goal_frames - self.total_frames
        if fps < 1.0:
            eta = 1
            print("Warning: FPS is less than 1.")
        else:
            eta = remaining / fps
        percent_done = int(100*(1.0 - (remaining/self.goal_frames)))


        print("===== Time Report =====================", flush=True)
        print("    Uptime:", self.seconds_to_string(int(uptime)), flush=True)
        print("    Frames:", self.thou(int(self.total_frames)), "/", self.thou(int(self.goal_frames)), "("+str(percent_done)+"%)", flush=True)
        print("    Frames Remaining:", self.thou(int(remaining)), flush=True)
        print("    FPS:", self.thou(int(fps) ) , flush=True)
        print("    Time Remaining:", self.seconds_to_string(int(eta)), flush=True)
        print("")