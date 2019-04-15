import numpy as np
import matplotlib.pyplot as plt

T = 20 #The period of the first sine in number of samples
NPeriods = 10 #How many periods to go through, relative to the faster sinusoid
N = T*NPeriods*3 #The total number of samples
t = np.arange(N) #Time indices

#Make the harmonic signal cos(t) + cos(3t)
xH = np.cos(2*np.pi*(1.0/T)*t) + np.cos(2*np.pi*(1.0/(3*T)*t))

fig = plt.figure()
#axFunction = fig.add_subplot(211)
#axFunction.plot(xH)


P1 = np.abs(np.fft.fft(xH))**2
#axFFT = fig.add_subplot(212)
#axFFT.plot(np.arange(len(P1)), P1)
#axFFT.set_xlim([0, 100])


axFunction = fig.add_subplot(211)
P1[15:100] = [0] * (100-15)
P1[560:] = [0] * len(P1[560:])
invFFT = np.fft.ifft(P1)
axFunction.plot(invFFT)


axFFT = fig.add_subplot(212)
axFFT.plot(np.arange(len(P1)), P1)
#axFFT.set_xlim([0, 100])
plt.show()