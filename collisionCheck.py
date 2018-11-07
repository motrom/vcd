#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 6/18/18

approach:
    find differences in position and angle of rectangles
    use simple version of separating line theorem for convex polygons:
        if all corners in (rotated) x axis are past other rectangle's length
        or all corners in y axis are past other rectangle's width
   the symmetry of rectangles allows for some speedups on top of this
"""
from math import cos, sin, hypot
import numpy as np
import numba

Vlen = 2.5
Vwid = 1.
maxdistance = hypot(Vlen, Vwid) * 2

@numba.jit(numba.b1(numba.f8[:], numba.f8[:]), nopython=True)
def collisionCheck(veh1, veh2):
    off_x = veh2[0] - veh1[0]
    off_y = veh2[1] - veh1[1]
    c1 = cos(veh1[2])
    s1 = sin(veh1[2])
    c2 = cos(veh2[2])
    s2 = sin(veh2[2])
    # because given point is front and center
    off_x += c1*Vlen - c2*Vlen
    off_y += s1*Vlen - s2*Vlen
    
    if hypot(off_x, off_y) > maxdistance: return False
    
    off_c = abs(c1*c2+s1*s2)
    off_s = abs(c1*s2-s1*c2)
    
    if abs(c1*off_x + s1*off_y) - off_c*Vlen - off_s*Vwid > Vlen:
        return False
    
    if abs(s1*off_x - c1*off_y) - off_s*Vlen - off_c*Vwid > Vwid:
        return False
    
    if abs(c2*off_x + s2*off_y) - off_c*Vlen - off_s*Vwid > Vlen:
        return False
    
    if abs(s2*off_x - c2*off_y) - off_s*Vlen - off_c*Vwid > Vwid:
        return False
    
    return True



def check(veh1, veh2, vlen = Vlen, vwid = Vwid):
    off_x = veh2[...,0] - veh1[...,0]
    off_y = veh2[...,1] - veh1[...,1]
    c1 = np.cos(veh1[...,2])
    s1 = np.sin(veh1[...,2])
    c2 = np.cos(veh2[...,2])
    s2 = np.sin(veh2[...,2])
    # because given point is front and center
    off_x += c1*vlen - c2*vlen
    off_y += s1*vlen - s2*vlen
    
    off_c = abs(c1*c2+s1*s2)
    off_s = abs(c1*s2-s1*c2)
    
    return (abs(c1*off_x + s1*off_y) < vlen + off_c*vlen + off_s*vwid) &\
           (abs(s1*off_x - c1*off_y) < vwid + off_s*vlen + off_c*vwid) &\
           (abs(c2*off_x + s2*off_y) < vlen + off_c*vlen + off_s*vwid) &\
           (abs(s2*off_x - c2*off_y) < vwid + off_s*vlen + off_c*vwid)
