import matplotlib.pyplot as plt
import numpy as np
import re

cpu = 'i5-4200U @ 1.60GHz'
gpu = 'GEForce GT 740M @ 810 MHz'

with open('./benchmark.log') as f:
    content = f.readlines()

def integer(str):
	num = re.search('([0-9].[0-9]*)e\+([0-9]*)',str)
	return float(num.group(1)) * 10**float(num.group(2))

content = [x.split() for x in content[1:]] 
# print content
plotter = []
for j in content:
	plotter.append([integer(i) for i in j])

plotter = np.array(plotter)

with open('./benchmark_cuda.log') as f2:
    content2 = f2.readlines()

content2 = [x.split() for x in content2[1:]] 
# print content
plotter2 = []
for j in content2:
	plotter2.append([integer(i) for i in j])

plotter2 = np.array(plotter2)

plt.semilogx(plotter[:,0],plotter[:,1],'o-',plotter2[:,0],plotter2[:,1],'o-')
plt.legend([cpu,gpu])
plt.show()
