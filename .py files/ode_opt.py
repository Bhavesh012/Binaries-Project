# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:58:27 2020

@author: BHavesh Rajpoot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 
from astropy import constants as ac, units as u
from prettytable import PrettyTable

def ode_opt(m1,m2,ti,tf,x0,y0,vx0,vy0):
    """This function solves the two-body odes using scipy.integrate.solve_ivp using,
    m - mass of stars in solar mass
    ti - initial time in year
    tf - final time in year
    x0, y0 - initial coordinates of the star in AU
    vx0, vy0 - initial velocity of the star in Km/s
    """
    
    #Universal Gravitional Constant
    G = ac.G.to('(AU**3)/(solMass year**2)').value
    
    #initial values
    x0 = x0*u.AU        #initial position
    y0 = y0*u.AU
    vx0 = vx0*u.km/u.s  #initial velocity 
    vy0 = vy0*u.km/u.s
    
    init_values = [x0.value, y0.value, vx0.to('AU/year').value, vy0.to('AU/year').value]
    
    t_span = (ti, tf)  # array for integration time t0 to tf
    t = np.linspace(ti, tf, 100000)

    def ode_func(t,init):
        """This function generates differential equations of 1 star in 2-body problem system from CoM frame in cartessian system."""
        x = init[0] #assigning the values
        y = init[1]
        vx = init[2]
        vy = init[3]
        r = np.sqrt(x*x + y*y) 

        M = (m2*m2*m2) / ((m1+m2)**2) #mass function

        k = G*M #universal constant

        d1 = (x) / (r*r*r) 
        d2 = (y) / (r*r*r)
        ax = -k*d1 
        ay = -k*d2
        return vx,vy,ax,ay 

    
    #ode solver - method RK4
    def ode_solver(ode_func, t_span, init_values, t):
        ode_sol = solve_ivp(ode_func, t_span, init_values,  method='RK45', t_eval = t)

        time = ode_sol.t      #time array
        sol = ode_sol.y       #solution array (consists arrays of x,y,vx,vy)
        return time,sol


    def orb_param(sol):
        """This function generates orbital parameters from the ode solution"""
        x1 = sol[0]*u.au
        y1 = sol[1]*u.au
        r1 = np.sqrt(x1*x1 + y1*y1)         # in AU

        m_ratio = m1/m2     #mass ratio
        m_sum = m1+m2       #mass sum

        x2 = -1*x1*m_ratio
        y2 = -1*y1*m_ratio
        r2 = np.sqrt(x2*x2 + y2*y2)         # in AU

        vx1 = sol[2]*u.AU/u.year
        vy1 = sol[3]*u.AU/u.year

        v1 = (np.sqrt(vx1*vx1 + vy1*vy1)).to('km/s').value  #orbital velocity of star 1 in km/s
        v2 = v1*m_ratio                            #orbital velocity of star 2 in km/s

        #calculating orbital parameters
        x1_min, x1_max = min(x1), max(x1) #periastron and apastron
        y1_min, y1_max = min(y1), max(y1)
        
        a1 = (x1_max - x1_min)/2  #semi-major axis of star 1
        b1 = (y1_max - y1_min)/2  #semi-minor axis of star 2
        e = np.sqrt(1-(b1.value**2/a1.value**2))  #eccentricity

        a2 = a1*m_ratio #semi-major axis of star 1

        p = 2*np.pi*np.sqrt((a1+a2)**3/(G*m_sum))  #time-period of Star 1 in year
        
        #printing tabular output
        t = PrettyTable()
        t.field_names = ["Parameter", "Value", "Unit"]
        t.align["Parameter"] = 'l'
        t.add_row(['Orbital Eccentricity',e,""])
        t.add_row(['Orbital Eccentricity',e,""])
        t.add_row(['Orbital Period',p.value,u.year])
        t.add_row(['Star 1: Semi-Major Axis',a1.value,u.au])
        t.add_row(['Star 2: Semi-Major Axis',a2.value,u.au])
        print(t)
        
        return r1,r2,x1,y1,x2,y2,v1,v2,a1,a2,e,m_ratio
    
    
    #plotting the curves
    def orb_pos_plot(r1,r2,time):
        """This function plots position vector curve."""
        plt.figure(figsize=(10,6))
        plt.plot(time, r1, label = 'Star 1')
        plt.plot(time, r2, label = 'Star 2')
        plt.title('Position Vector Curve')
        plt.ylabel("Position Vector, r (in AU)")
        plt.xlabel("Time (in years)")
        plt.legend()
        plt.grid()
        plt.show()
        
    def orb_vel_plot(v1,v2,time):
        """This function plots orbital velocity curve."""
        plt.figure(figsize=(10,6))
        plt.plot(time, v1, label = 'Star 1')
        plt.plot(time, v2, label = 'Star 2')
        plt.title('Orbital Velocity Curve')
        plt.ylabel("Orbital Velocity (in km/s)")
        plt.xlabel("Time (in years)")
        plt.legend()
        plt.grid()
        plt.show()
        
    
    def mark(sol):
        """"This function generates markers for plotting."""
        X1 = sol[0,25000] #in AU
        Y1 = sol[1,25000] #in AU
        X2 = -1*X1*m_ratio
        Y2 = -1*Y1*m_ratio
        return X1,Y1,X2,Y2
    
    #plotting the orbit
    def orb_plot(x1,x2,X1,Y2,a1,a2,e):
        """This function plots orbit curves."""
        plt.figure(figsize=(12,8))
        plt.plot(x1, y1, color='red', label='Star 1 Orbit')
        plt.plot(x2, y2, color='blue', label='Star 2 Orbit')
        plt.plot(0, 0, 'kx', label = 'CoM')
        plt.plot(2*a1*e, 0, 'k.')
        plt.plot(-2*a2*e, 0, 'k.')
        plt.plot(X1, Y1, color='red', marker = 'o', markersize=2*10, label='Star 1')
        plt.plot(X2, Y2, color='blue', marker = 'o', markersize=1*10, label='Star 2')
        plt.plot([X1, X2], [Y1, Y2], 'k--')
        plt.title('Orbit Plot')
        plt.xlim()
        plt.ylim()
        plt.legend(labelspacing=1.5)
        plt.grid()
        plt.show()
    
    time,sol = ode_solver(ode_func, t_span, init_values, t)
    r1,r2,x1,y1,x2,y2,v1,v2,a1,a2,e,m_ratio = orb_param(sol)
    orb_pos_plot(r1,r2,time)
    orb_vel_plot(v1,v2,time)
    X1,Y1,X2,Y2 = mark(sol)    
    orb_plot(x1,x2,X1,Y2,a1,a2,e)
    
    
ode_opt(6,2,0,2.82,-0.1,0,0,-45.89)
