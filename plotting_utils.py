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

# These are some helpful utilities to grab data from a tensorboard log and 
# plot it with matplotlib (or however you like to plot things in python). 
# This was used to make the plots in the README.

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.signal import lfilter

def rolling_window(x, y, window_length=10):
    """ Rolling window average """
    rolled = np.convolve(y, np.ones(window_length), 'valid') / window_length
    return x[window_length-1:], rolled

def ema(y, alpha=0.95):
    """ Exponential moving average (tb style smoothing) """
    zi = [y[0]] # seed the filter state with first value
    # filter can process blocks of continuous data if <zi> is maintained
    y_ema, zi = lfilter([1.-alpha], [1., -alpha], y, zi=zi)
    return y_ema

def get_tags(folder_name):
    """ Get all available tags in the logs """
    return EventAccumulator(folder_name).Reload().Tags()['scalars']

def tb_load(folder_name, tag):
    """  Collect scalar data from a list of files in a folder """
    data = EventAccumulator(folder_name).Reload().Scalars(tag)
    x = [event.step for event in data]
    y = [event.value for event in data]
    return x, y

