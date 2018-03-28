import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
import pandas as pd
import numpy as np

#or: points red, -b: line blue
plt.plot([1,2,3,4], [1,4,9,16], 'or')
plt.plot([1,2,3,4], [1,4,9,16], '-b')
plt.ylabel('some numbers')
#x and y range
plt.axis([0,6,0,20])
plt.show()

t = np.arange(0., 5., 0.2)
t1 = np.arange(0., 5., 0.1)
t2 = np.arange(0., 5., 0.02)

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#with coma after line we use tuple unpacking to get only the first element of the list
line, = plt.plot(t, t**2, '-', linewidth = 2, color = 'r')
line.set_antialiased(False)
#other way to set the line parameters
plt.setp(line, color = 'r', linewidth = 5)

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo')

plt.subplot(212)
plt.plot(t2, f(t2), 'r^')

mu, sigma = 100, 15
x = mu + sigma*np.random.randn(1000)

n, bins, patches = plt.hist(x, 50, normed=1, facecolor = 'm', alpha = 0.75)
plt.title = 'Histogram of random numbers'
plt.text(60, 0.02, r'$\mu=100, \ \ sigma=15$')


#arrows with annotations
s = np.cos(2*np.pi*t2)
line, = plt.plot(t2,s, lw = 2)
plt.axis([0,10,-2,2])
#arrow: xy: location (x,y) for arrow end/tip, xytext: location of text
plt.annotate('Maximum', xy=(2,1), xytext=(3,1.5), arrowprops=dict(facecolor ='black', shrink = 0.05),)

#logarythmic scale
plt.yscale('log')

#bad #strings are inmutable in python, slow and inefficient
nums = ""
for n in range(20):
    nums += str(n)
print nums

#better # accumlates the parts of desired string in a list which is mutable and then join it with append()
nums = []
for n in range(20):
    nums.append(str(n))
print "".join(nums)

#better #list comprahension are faster for this
nums = [str(n) for n in range(20)]
print "".join(nums)

#best # maps a function 'str' to iterable (range(20)), can be even faster than comprahension list in some cases
nums = map(str, range(20))
print "".join(nums)

#two y axis with two different scales
plt.plot(df_gdp.query('at_ds_location == @i').at_dt_year,
         df_gdp.query('at_ds_location == @i').gdp_growth_perc, 'or') 
plt.plot(df_pax_airport.query('at_ds_country == @i').at_dt_year,
         df_pax_airport.query('at_ds_country == @i').mt_pax_country_growth, 'ob') 
plt.xlim([2010.5,2017.5])

fig, ax1 = plt.subplots()
ax1.plot(df_gdp.query('at_ds_location == @i').at_dt_year,
         df_gdp.query('at_ds_location == @i').mt_gdp, 'or') 
ax1.set_ylabel('gdp', color='b')
ax1.tick_params('y', colors='b')
ax1.set_xlabel('Time (years)')
ax2 = ax1.twinx()
ax2.plot(df_pax_airport.query('at_ds_country == @i').at_dt_year,
         df_pax_airport.query('at_ds_country == @i').mt_pax_country, 'ob')
ax2.set_ylabel('pax', color='r')
ax2.tick_params('y', colors='r')
plt.xlim([2010.5,2017.5])
fig.tight_layout()
plt.show()

#like above but two subplots with second scale on y axis
   # create all axes we need
    ax1 = plt.subplot(211)
    ax2 = ax1.twinx()
        
    #fig, ax1 = plt.subplots()
    ax1.plot(df_gdp.query('at_ds_location == @i').at_dt_year,
             df_gdp.query('at_ds_location == @i').mt_gdp, 'or') 
    ax1.set_ylabel('gdp', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('Time (years)')
    #ax2 = ax1.twinx()
    ax2.plot(df_pax_airport.query('at_ds_country == @i').at_dt_year,
             df_pax_airport.query('at_ds_country == @i').mt_pax_country, 'ob')
    ax2.set_ylabel('pax', color='r')
    ax2.tick_params('y', colors='r')
    
    plt.xlim([2010.5,2017.5])

    #fig.tight_layout()
    ax3 = plt.subplot(212)
    ax4 = ax3.twinx()
    ax3.plot(df_gdp.query('at_ds_location == @i').at_dt_year,
             df_gdp.query('at_ds_location == @i').gdp_growth_perc, 'or') 
    ax3.plot(df_pax_airport.query('at_ds_country == @i').at_dt_year,
             df_pax_airport.query('at_ds_country == @i').mt_pax_country_growth, 'ob') 
    #plt.xlim([2010.5,2017.5])
    plt.xlim([2010.5,2017.5])
    plt.show()