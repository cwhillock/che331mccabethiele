import math
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as mpl

class compound:
    def __init__(self,A,B,C,ma):
        self.A = A
        self.B = B
        self.C = C
        self.marguleA = ma
    gamma = 0
    ppressure=0
    X = 0
    Y = 0

def setGamma(cp1,cp2):
    'sets the activity coefficient of two compounds based on the internet variables for X and A'
    'input compound 1, compound 2'
    cp1.gamma = math.exp( math.pow(cp2.X,2) * (cp1.marguleA + 2 * (cp2.marguleA - cp1.marguleA) * cp1.X))
    cp2.gamma = math.exp( math.pow(cp1.X,2) * (cp2.marguleA + 2 * (cp1.marguleA - cp2.marguleA) * cp2.X))

def antoine(cp,t):
    'return partial pressure of a compound'
    'input compound, temperature'
    exp = cp.A - cp.B / (cp.C + t)
    return math.pow(10,exp) / 7.5006

def BPP(cp1,cp2,t,x1):
    'calculates a bubble point pressure calculation'
    'input compound 1, compound 2, temperature, liquid mol fraction of compound 1'
    cp1.X = x1
    cp2.X = 1-x1
    setGamma(cp1,cp2)
    cp1.ppressure = antoine(cp1,t)
    cp2.ppressure = antoine(cp2,t)
    tpressure = cp1.X * cp1.gamma * cp1.ppressure + cp2.X * cp2.gamma * cp2.ppressure
    cp1.Y = cp1.X * cp1.gamma * cp1.ppressure / tpressure
    cp2.Y = cp2.X * cp2.gamma * cp2.ppressure / tpressure
    return tpressure

def BPT(cp1,cp2,x1,P,Tguess,imax):
    'performs a bubble point temperature calculation'
    'input compound 1, compound 2, liquid mol fraction of compound 1, pressure, a temperature guess, and max iterations'
    Tmin = 0
    cp1.X = x1
    cp2.X = 1-x1
    setGamma(cp1,cp2)

    for i in range(imax):
        cp1.ppressure = antoine(cp1,Tguess)
        cp2.ppressure = antoine(cp2,Tguess)
        TPcalc = (cp1.X * cp1.gamma * cp1.ppressure) + (cp2.X * cp2.gamma * cp2.ppressure)
        obj = TPcalc - P
        
        if abs(obj) < 0.001:
            print("massive success!")
            Tmin = Tguess
            break

        if obj > 0:
            Tguess = (Tguess + Tmin) / 2

        else:
            if Tguess > Tmin:
                Tmin = Tguess
            Tguess = 1.5 * Tguess
    cp1.ppressure = antoine(cp1,Tmin)
    cp2.ppressure = antoine(cp2,Tmin)
    cp1.Y = (cp1.X * cp1.gamma * cp1.ppressure) / P
    cp2.Y = (cp2.X * cp2.gamma * cp2.ppressure) / P
    print(obj,x1)
    return Tmin

def genTXY(cp1,cp2,P,stepsize,Tguess,imax):
    'does a series of BPT calculations over mol fractions from 0 to 1'
    'input compound 1,compound 2, pressure, stepsize between mol fractions to calcualte, a temperature guess over the whole range, and max iterations'
    'outputs a 2d array with the data in order of x1,x2,gamma1,gamma2,T,Psat1,Psat2,Y1'
    temp = int(1/stepsize + 1)
    valtemp = np.empty([temp,8])
    for k in range(temp):
        xtemp = k * stepsize
        valtemp[k,4] = round(BPT(cp1,cp2,xtemp,P,Tguess,imax),2)
        valtemp[k,0] = round(cp1.X,3)
        valtemp[k,1] = round(cp2.X,3)
        valtemp[k,2] = round(cp1.gamma,6)
        valtemp[k,3] = round(cp2.gamma,6)
        valtemp[k,5] = round(cp1.ppressure,2)
        valtemp[k,6] = round(cp2.ppressure,2)
        valtemp[k,7] = round(cp1.Y,3)
    return valtemp

def getXfromY(yval,xarr,yarr):
    'input an array of y values, array of x values, and y value to evaluate'
    'linearly interpolates a y value between equal sets of x and y data'
    'outputs an x value that would correspond to the given y'
    bindex = 0
    while yarr[bindex] <= yval:
        bindex += 1
    fracdiff = (yval - yarr[bindex]) / (yarr[bindex+1] - yarr[bindex])
    newx = (xarr[bindex+1] - xarr[bindex]) * fracdiff + xarr[bindex]
    return newx

def operatingfunction(oparray,xval):
    'input operating specifications, x value to evaluate'
    'creates a piecewise function from operating specifications'
    'includes stripping line and rectifying line, y=x for all others'
    qlineterm1 = oparray[3] / (oparray[3] - 1)
    qlineterm2 = oparray[1] / (oparray[3]-1)
    rlineterm1 = oparray[4] / (oparray[4]+1)
    rlineterm2 = oparray[0] / (oparray[4]+1)
    crosspoint = (rlineterm2 + qlineterm2) / (qlineterm1 - rlineterm1)
    if xval > oparray[2]:
        if xval > crosspoint:
            if xval > oparray[0]:
                return xval
            return rlineterm1 * xval + rlineterm2
        crossy = rlineterm1 * crosspoint + rlineterm2
        slope = (crossy - oparray[2]) / (crosspoint - oparray[2])
        return slope * xval + oparray[2] * (1-slope)
    return xval

def getOperatingPoints(oparray):
    'input operating specifications'
    'creates a set of 3 data points for use in graphing the operating line'
    qlineterm1 = oparray[3] / (oparray[3] - 1)
    qlineterm2 = oparray[1] / (oparray[3]-1)
    rlineterm1 = oparray[4] / (oparray[4]+1)
    rlineterm2 = oparray[0] / (oparray[4]+1)
    crosspoint = (rlineterm2 + qlineterm2) / (qlineterm1 - rlineterm1)
    crossy = rlineterm1 * crosspoint + rlineterm2
    opvalues = np.array([[oparray[0],oparray[0]],[crosspoint,crossy],[oparray[2],oparray[2]]])
    return opvalues

def drawStageLines(eqx,eqy,oparray):
    'input equilibrium data x values, eq data y values, and operating specifications'
    'draws the stage lines from top to bottom between equilibrium data and operating line'
    xcursor = oparray[0]
    ycursor = oparray[0]
    xvalues = [xcursor]
    yvalues = [ycursor]
    while xcursor > oparray[2]:
        xcursor = getXfromY(ycursor,eqx,eqy)
        xvalues.append(xcursor)
        yvalues.append(ycursor)
        ycursor = operatingfunction(oparray,xcursor)
        xvalues.append(xcursor)
        yvalues.append(ycursor)
    length = len(xvalues)
    stageLines = np.empty([length,2])
    for i in range(length):
        stageLines[i,0] = xvalues[i]
        stageLines[i,1] = yvalues[i]
    return stageLines

ethanol = compound(8.13484,1662.48,238.131,1.2713)
DCE = compound(7.29525,1407.85,235.48,0.5955)
'defining compounds in this system'

header = np.array(["x1","x2","gamma1","gamma2","T","P1sat","P2sat","y1"])
values = genTXY(ethanol,DCE,101.325,0.01,79,100)
combined = np.vstack((header,values))
print(tabulate(combined))
'generating and printing a TXY table'
'the header and numbers are in 2 separate arrays then joined'

mpl.plot(values[:,0],values[:,4],values[:,7],values[:,4])
mpl.show()
'creating a txy plot'

mpl.plot(values[:,0],values[:,7])
mpl.plot([0,1],[0,1],ls = '--')
mpl.show()
'creating an xy plot'

xvals = np.array(values[:,0])
yvals = np.array(values[:,7])
'creating 2 separate arrays for the x data points and y data points'

operatingSpecifications = np.array([0.4,0.25,0.05,0.4,1.5])
'order: Xd,Z,Xb,q,R'
'defining operating specs'

testeqdatax = np.array([1,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.08,0.06,0.04,0.02,0])
testeqdatay = np.array([1,0.979,0.958,0.915,0.87,0.825,0.779,0.729,0.665,0.579,0.517,0.418,0.365,0.304,0.23,0.134,0])
'test values not used'

opvalues = getOperatingPoints(operatingSpecifications)
stageLines = drawStageLines(xvals,yvals,operatingSpecifications)
'creating 2 arrays containing the data points for the operating line and stagelines'
'stage lines come in a 2d array of x and y values'

mpl.plot(xvals,yvals)
mpl.plot(opvalues[:,0],opvalues[:,1])
mpl.plot(stageLines[:,0],stageLines[:,1])
mpl.plot([0,1],[0,1], ls = "--")
mpl.show()
'plotting eq data with y=x line, operating line, and stage line'








