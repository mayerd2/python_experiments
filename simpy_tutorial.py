# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:38:52 2025

@author: dirkm
"""

def car(env):

    while True:

        print('Start parking at %d' % env.now)

        parking_duration = 5

        yield env.timeout(parking_duration)

        print('Start driving at %d' % env.now)

        trip_duration = 2

        yield env.timeout(trip_duration)
        
import simpy

env = simpy.Environment()

env.process(car(env))

env.run(until=15)