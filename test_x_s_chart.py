import unittest

import collections
import math
import json

import numpy as np

import obspy

from x_s_chart import ni_factorial, percent_difference
from x_s_chart import c4, control_limit_constants
from x_s_chart import S_chart


class MathTests(unittest.TestCase):
    

    def test_percent_difference(self):
        d = percent_difference(5,4)
        self.assertTrue(d >= 0.0 and d <= 100.0 )
        d = percent_difference(4,5)
        self.assertTrue(d >= 0.0 and d <= 100.0 )
        return

    def test_ni_factorial_w_ni(self):
        ni = 3.5
        self.assertTrue( percent_difference(11.632, ni_factorial(ni)) < 0.1 )
        return

    def test_ni_factorial_w_m(self):
        m = 12
        self.assertEqual(math.factorial(m),ni_factorial(m))
        return

    def test_c4_calculation(self):
        m = 10
        self.assertTrue( percent_difference(0.9727, c4(m)) < 0.1) 
        return

    def test_control_limit_constants(self):
        m = 10
        A3, B3, B4 = control_limit_constants(m, calculate=True)
        self.assertTrue(percent_difference(0.975,A3) < 1)
        self.assertTrue(percent_difference(0.284,B3) < 1)
        self.assertTrue(percent_difference(1.716,B4) < 1)
        return

class S_chartTests(unittest.TestCase):

    def setUp(self):
        #json.JSONDecoder(object_pairs_hook=collections.OrderedDict)
        metric = u"sample_mean"
        filename = "test_data/UW.GNW...BHZ_sample_mean.json"
        with open(filename) as fp:
            d = json.load(fp, object_pairs_hook=collections.OrderedDict)
        m = d[u"measurements"][metric]
        data = []
        for v in m:
            data.append(v["value"])
        self.my_chart = S_chart(data,winlength=7,m=10,description=metric)
        return

    def test_instantiate_large_array(self):
        d = np.random.normal(2.0,0.45,10000)
        s_chart = S_chart(d,m=15)
        print(s_chart)
        return

    def test_remainder(self):
        d = np.random.normal(2.0,0.45,10000)
        s_chart = S_chart(d,winlength=350,m=15)
        #print(s_chart)
        return

    def test_resize_m(self):
        d = np.random.normal(2.0,0.45,1000)
        s_chart = S_chart(d,m=15)
        self.assertEqual(s_chart.m,10)
        return

    def test_print(self):
        print(self.my_chart)
        return

    def test_plot1(self):
        self.my_chart.plot()
        return

    def test_plot2(self):
        d = np.random.normal(2.0,0.25,10000)
        s_chart = S_chart(d,winlength=7,m=5)
        s_chart.description = "10000 samples from a normal distribution with mean 2"
        s_chart.plot()
        return

if __name__ == "__main__":
    unittest.main()
