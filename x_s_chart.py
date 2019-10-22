import math

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import obspy

CONSTANTS = namedtuple("CONSTANTS",["A", "A2", "A3", "c4", "B3", "B4", "B5", "B6", "d2", "inv_d2", "d3", "D1", "D2", "D3", "D4"])
TABLE =  { 
  2 : CONSTANTS(2.121, 1.880, 2.659, 0.7979, 0.000, 3.267, 0.000, 2.606, 1.128, 0.8862, 0.853, 0.000, 3.686, 0.000, 3.267),
  3 : CONSTANTS(1.732, 1.023, 1.954, 0.8862, 0.000, 2.568, 0.000, 2.276, 1.693, 0.5908, 0.888, 0.000, 4.358, 0.000, 2.575),
  4 : CONSTANTS(1.500, 0.729, 1.628, 0.9213, 0.000, 2.266, 0.000, 2.088, 2.059, 0.4857, 0.880, 0.000, 4.698, 0.000, 2.282),
  5 : CONSTANTS(1.342, 0.577, 1.427, 0.9400, 0.000, 2.089, 0.000, 1.964, 2.326, 0.4299, 0.864, 0.000, 4.918, 0.000, 2.114),
  6 : CONSTANTS(1.225, 0.483, 1.287, 0.9515, 0.030, 1.970, 0.029, 1.874, 2.534, 0.3946, 0.848, 0.000, 5.079, 0.000, 2.004),
  7 : CONSTANTS(1.134, 0.419, 1.182, 0.9594, 0.118, 1.882, 0.113, 1.806, 2.704, 0.3698, 0.833, 0.205, 5.204, 0.076, 1.924),
  8 : CONSTANTS(1.061, 0.373, 1.099, 0.9650, 0.185, 1.815, 0.179, 1.751, 2.847, 0.3512, 0.820, 0.388, 5.307, 0.136, 1.864),
  9 : CONSTANTS(1.000, 0.337, 1.032, 0.9693, 0.239, 1.761, 0.232, 1.707, 2.970, 0.3367, 0.808, 0.547, 5.394, 0.184, 1.816),
 10 : CONSTANTS(0.949, 0.308, 0.975, 0.9727, 0.284, 1.716, 0.276, 1.669, 3.078, 0.3249, 0.797, 0.686, 5.469, 0.223, 1.777),
 11 : CONSTANTS(0.905, 0.285, 0.927, 0.9754, 0.321, 1.679, 0.313, 1.637, 3.173, 0.3152, 0.787, 0.811, 5.535, 0.256, 1.744),
 12 : CONSTANTS(0.866, 0.266, 0.886, 0.9776, 0.354, 1.646, 0.346, 1.610, 3.258, 0.3069, 0.778, 0.923, 5.594, 0.283, 1.717),
 13 : CONSTANTS(0.832, 0.249, 0.850, 0.9794, 0.382, 1.618, 0.374, 1.585, 3.336, 0.2998, 0.770, 1.025, 5.647, 0.307, 1.693),
 14 : CONSTANTS(0.802, 0.235, 0.817, 0.9810, 0.406, 1.594, 0.399, 1.563, 3.407, 0.2935, 0.763, 1.118, 5.696, 0.328, 1.672),
 15 : CONSTANTS(0.775, 0.223, 0.789, 0.9823, 0.428, 1.572, 0.421, 1.544, 3.472, 0.2880, 0.756, 1.203, 5.740, 0.347, 1.653),
 16 : CONSTANTS(0.750, 0.212, 0.763, 0.9835, 0.448, 1.552, 0.440, 1.526, 3.532, 0.2831, 0.750, 1.282, 5.782, 0.363, 1.637),
 17 : CONSTANTS(0.728, 0.203, 0.739, 0.9845, 0.466, 1.534, 0.458, 1.511, 3.588, 0.2787, 0.744, 1.356, 5.820, 0.378, 1.622),
 18 : CONSTANTS(0.707, 0.194, 0.718, 0.9854, 0.482, 1.518, 0.475, 1.496, 3.640, 0.2747, 0.739, 1.424, 5.856, 0.391, 1.609),
 19 : CONSTANTS(0.688, 0.187, 0.698, 0.9862, 0.497, 1.503, 0.490, 1.483, 3.689, 0.2711, 0.733, 1.489, 5.889, 0.404, 1.596),
 20 : CONSTANTS(0.671, 0.180, 0.680, 0.9869, 0.510, 1.490, 0.504, 1.470, 3.735, 0.2677, 0.729, 1.549, 5.921, 0.415, 1.585),
 21 : CONSTANTS(0.655, 0.173, 0.663, 0.9876, 0.523, 1.477, 0.516, 1.459, 3.778, 0.2647, 0.724, 1.606, 5.951, 0.425, 1.575),
 22 : CONSTANTS(0.640, 0.167, 0.647, 0.9882, 0.534, 1.466, 0.528, 1.448, 3.819, 0.2618, 0.720, 1.660, 5.979, 0.435, 1.565),
 23 : CONSTANTS(0.626, 0.162, 0.633, 0.9887, 0.545, 1.455, 0.539, 1.438, 3.858, 0.2592, 0.716, 1.711, 6.006, 0.443, 1.557),
 24 : CONSTANTS(0.612, 0.157, 0.619, 0.9892, 0.555, 1.445, 0.549, 1.429, 3.895, 0.2567, 0.712, 1.759, 6.032, 0.452, 1.548),
 25 : CONSTANTS(0.600, 0.153, 0.606, 0.9896, 0.565, 1.435, 0.559, 1.420, 3.931, 0.2544, 0.708, 1.805, 6.056, 0.459, 1.541) }

def times(array,dt=1,unit="day"):
    """
        Each point in data represents dt (unit) time period
    """
    nmax = len(array)
    remainder = nmax % dt
    nmax = int(nmax/dt)
    times = range(0,nmax*dt+remainder)
    return times

def percent_difference(value1,value2):
    return 100*abs(value1-value2)/value1 

def ni_factorial(n):
    """ 
        non-integer factorial
    """
    from scipy.special import factorial
    if type(n) == int or n.is_integer():
        return factorial(int(n))
    else:
        ni = n
        f = ni
        while ni > 0.5:
            f = f * (ni-1)
            ni = ni - 1
        f = f * math.sqrt(math.pi)
    return f

def c4(m):
    """
        c4 factor, see:
        http://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm#C4 
    """
    
    if m <= 1:
        raise ValueError("Error: cannot calculate c4, number of samples has to be greater than 1")
    numerator = ni_factorial(m/2.0 - 1)
    denominator = ni_factorial((m-1)/2.0 - 1)
    c = math.sqrt( 2.0/(m-1) )
    c4 = c * (numerator/denominator)
    return c4
    
def control_limit_constants(m, sigma=3, calculate=False):
    """
        return B4 and B3 factors that can be multiplied with the mean sample standard deviation to get
        the Upper Control Limit and Lower Control Limit for sample standard deviation. 
        return A3, factor to find ucl and lcl for X-chart.
    """
    if calculate:
        try:
            constant = sigma/c4(m) * math.sqrt( 1 - c4(m)*c4(m) )
        except:
            raise
        B4 = 1.0 + constant
        B3 = 1.0 - constant
        A3 = 3/(c4(m)*math.sqrt(m))
        return A3, B3, B4
    else:
        return TABLE[m].A3, TABLE[m].B3, TABLE[m].B4

def plot_chart(data,dt,ucl,lcl,ax,title="Chart"):
        ax.set_title(title)
        ax.plot(times(data,dt=dt),data,'k-o')
        npoints = len(self.sarray)
        ax2.plot((0,npoints),(self.mean_std,self.mean_std),'b-')
        ax2.plot((0,npoints),(self.s_lcl,self.s_lcl),'r-')
        ax2.plot((0,npoints),(self.s_ucl,self.s_ucl),'r-')
        text = "mean_std = {:.0f}".format(self.mean_std)
        ax2.annotate(text, xy=(len(self.sarray),self.mean_std),xytext=(0.8,0.25),xycoords="axes fraction")
        ax2.set_xlim([0,npoints])
        ymin = np.min(self.sarray)
        ymax = np.max(self.sarray)
        #ymin = self.mean_std - 2*abs(self.mean_std)
        #ymax = self.mean_std + 2*abs(self.mean_std)
        ax2.set_ylim(ymin,ymax)


class S_chart():
    """
        given obspy.Trace, np.ndarray, or array of floats, return the s-chart, lower-control-limit and upper-control-limit
        @param data array with measurements, type can be obspy.Trace, np.ndarray, or regular array (list)
        @param winlength number of data samples in each statistical sample
        @param m number of windows used to determine the average sample standard deviation
        @param control limits will be sigma number of standard deviations away from the mean
        @param description Used for plot titles
    """

    def __init__(self,data, winlength=100, m=10, sigma=3, description="Measured Data", **kwargs):

        if isinstance(data.__class__,obspy.core.trace.Trace):
            self.data = data.data
        else:
            self.data = data
        self.m = int(m)
        self.winlength = int(winlength)
        nmax = len(self.data)
        if self.m*self.winlength > nmax:
            # m is too big, change it to the max possible value given the data length
            self.m = int(nmax/self.winlength)
        self.description = description
        self.mean = self._calculate_mean()
        self.mean_std = self._calculate_mean_std()
        self.sigma = sigma
        self.A3, self.B3, self.B4 = control_limit_constants(m,sigma)
        self.sarray = self._get_s_statistic()
        self.marray = self._get_m_statistic()
        for k,v in kwargs:
            self.k = v 

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self,value):
        self._description = value
        return

    @property
    def mean_std(self):
        return self._mean_std

    @mean_std.setter
    def mean_std(self, value):
        self._mean_std = value
        return
    
    @property
    def s_ucl(self):
        return self.mean_std*self.B4

    @property
    def s_lcl(self):
        return self.mean_std*self.B3

    @property
    def x_ucl(self):
        return self.mean + self.mean_std*self.A3

    @property
    def x_lcl(self):
        return self.mean - self.mean_std*self.A3

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value
    
    @property
    def m(self):
        return self._m

    @m.setter
    def m(self,value):
        self._m = value

    @property
    def sarray(self):
        return self._sarray

    @sarray.setter
    def sarray(self, value):
        self._sarray = value

    @property
    def marray(self):
        return self._marray

    @marray.setter
    def marray(self, value):
        self._marray = value

    def _calculate_mean_std(self):
        average = 0.
        nmax = len(self.data)
        for i in range(0,self.m):
            istart = i*self.winlength
            iend = istart+self.winlength
            if iend > nmax:
                iend = nmax
            average = average + np.std(self.data[istart:iend],ddof=1)
        average = average/self.m
        return average
    
    def _calculate_mean(self):
        average = 0.
        nmax = len(self.data)
        for i in range(0,self.m):
            istart = i*self.winlength
            iend = istart+self.winlength
            if iend > nmax:
                iend = nmax
            average = average + np.mean(self.data[istart:iend])
        average = average/self.m
        return average

    def _get_s_statistic(self):
        self._sarray = []
        nmax = len(self.data)
        nwindows = int(nmax/self.winlength) + 1
        for i in range(0,nwindows):
            istart = i * self.winlength
            iend = istart + self.winlength
            if iend > nmax:
                iend = nmax
            self._sarray.append(np.std(self.data[istart:iend], ddof=0))
        self._sarray = np.array(self._sarray)
        return self._sarray

    def _get_m_statistic(self):
        self._marray = []
        nmax = len(self.data)
        nwindows = int(nmax/self.winlength) + 1
        for i in range(0,nwindows-1):
            istart = i * self.winlength
            iend = istart + self.winlength
            if iend > nmax:
                iend = nmax
            self._marray.append(np.mean(self.data[istart:iend]))
        self._marray = np.array(self._marray)
        return self._marray

    def __str__(self):
        message = """
                      {}-sigma S-chart (sample standard deviation)
                      -----------------------------------
                      data length = {}
                      window-length = {} time points
                      number of windows used to determine average standard deviation = {}
                      average data standard deviation = {}
                      average data mean = {}
                      Lower Control Limit for standard deviation = {} 
                      Upper Control Limit for standard deviation = {}
                      Lower Control Limit for mean = {} 
                      Upper Control Limit for mean = {}
                  """.format(self.sigma,len(self.data),self.winlength,self.m,self.mean_std,self.mean,
                      self.s_lcl,self.s_ucl,self.x_lcl,self.x_ucl)
        return message


    def plot(self,dt=1,unit="day"):
        fig = plt.figure(num=1, figsize=(8,10))
        ax1 = fig.add_subplot(311)
        ax1.set_title(self.description)
        ax1.plot(times(self.data),self.data,'k-')
        ax1.plot((0,len(self.data)),(self.mean,self.mean),'b-')
        text = "mean = {:.0f}".format(self.mean)
        ax1.annotate(text, xy=(len(self.data),self.mean),xytext=(0.8,0.25),xycoords="axes fraction")

        stimes = np.array(range(0,int(len(self.data)/(self.winlength*dt))))
        ax2 = fig.add_subplot(312)
        title = "Standard deviation of {} samples".format(self.winlength)
        ax2.set_title(title)
        ax2.plot(times(self.sarray,dt=self.winlength),self.sarray,'k-o')
        npoints = len(self.sarray)
        ax2.plot((0,npoints),(self.mean_std,self.mean_std),'b-')
        ax2.plot((0,npoints),(self.s_lcl,self.s_lcl),'r-')
        ax2.plot((0,npoints),(self.s_ucl,self.s_ucl),'r-')
        text = "mean_std = {:.0f}".format(self.mean_std)
        ax2.annotate(text, xy=(len(self.sarray),self.mean_std),xytext=(0.8,0.25),xycoords="axes fraction")
        ax2.set_xlim([0,npoints])
        ymin = np.min(self.sarray)
        ymax = np.max(self.sarray)
        #ymin = self.mean_std - 2*abs(self.mean_std)
        #ymax = self.mean_std + 2*abs(self.mean_std)
        ax2.set_ylim(ymin,ymax)

        ax3 = fig.add_subplot(313)
        title = "Mean value of {} values".format(self.winlength)
        ax3.set_title(title)
        ax3.plot(times(self.marray,dt=self.winlength),self.marray,'k-o')
        npoints = len(self.sarray)
        ax3.plot((0,npoints),(self.mean,self.mean),'b-')
        ax3.plot((0,npoints),(self.x_lcl,self.x_lcl),'r-')
        ax3.plot((0,npoints),(self.x_ucl,self.x_ucl),'r-')
        text = "mean = {:.0f}".format(self.mean)
        ax3.annotate(text, xy=(len(self.marray),self.mean),xytext=(0.8,0.25),xycoords="axes fraction")
        ax3.set_xlim([0,npoints])
        #ymin = self.mean - 2*abs(self.mean)
        #ymax = self.mean + 2 * abs(self.mean)
        ymin = np.min(self.marray)
        ymax = np.max(self.marray)
        ax3.set_ylim(ymin,ymax)

        plt.show()
