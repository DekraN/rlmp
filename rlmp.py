#######################################
####### Roofline Model Plotter ########
##### Marco Chiarelli @ CMCC ##########
## marco.chiarelli@cmcc.it 20/03/2019##
#######################################
#######################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sys import argv
    
ap = argparse.ArgumentParser()

ap.add_argument("-k", "--kern", "--kernel", metavar="KERNEL-NAME", type=str, required=True, help="your kernel name")
ap.add_argument("-m", "--mach", "--machine", metavar="MACHINE-NAME", type=str, required=True, help="your machine name")
ap.add_argument("-f", "--fops", "--flops", "--Flops", "--FLOPS", metavar="FLOPS", type=float, required=True, help="your kernel floating point operations (FLOPS)")
ap.add_argument("-l", "--l3", "--l3cm", "--L3CM", metavar="LL-CACHE-MISSES", type=int, required=True, help="your kernel last level cache misses")
ap.add_argument("-e", "-t", "--et", "--extime", "--exectime", metavar="EXECUTION-TIME", type=float, required=True, help="your kernel execution time")
ap.add_argument("-ll", "--l3l", "--l3cl", "--l3cls", "--l3cachelinesize", metavar="LL-CACHE-LINE-SIZE", type=int, required=True, help="last level cache line size")
ap.add_argument("-b", "--bm", "--bmax", metavar="MAXIMUM-BANDWIDTH", type=float, required=True, help="your machine maximum bandwidth")
ap.add_argument("-p", "--pp", "--peakperf", metavar="PEAK-FP-PERFORMANCE", type=float, required=True, help="your machine peak performance")
ap.add_argument("-fopt", "--fopt", "--flopsopt", "--Flopsopt", "--FLOPSopt", metavar="FLOPS", type=float, help="your optimized kernel floating point operations (FLOPS)")
ap.add_argument("-lopt", "--l3opt", "--l3cmopt", "--L3CMopt", metavar="LL-CACHE-MISSES", type=int, help="your optimized kernel last level cache misses")
ap.add_argument("-eopt", "-topt", "--etopt", "--extimeopt", "--exectimeopt", metavar="EXECUTION-TIME", type=float, help="your optimized kernel execution time")

ap.add_argument("-vm", "--vm", "--verbose_machine", action='store_true', help="enable verbosity for machine info")
ap.add_argument("-vk", "--vk", "--verbose_kernel", action='store_true', help="enable verbosity for kernel info")
ap.add_argument("-fo", "--out_file", metavar="OUTPUT-FILE", type=str, default="rlm.png", help="the image output file to save")
args = vars(ap.parse_args())

# Kernel Info

your_kernel_name = args["kern"]
your_machine_name = args["mach"]
Flops = args["fops"]
Gflops = Flops/(10**9)
L3CM = args["l3"]
etime = args["et"]
GFLOPS = Gflops/etime # y coordinate

global Flops_opt
global Gflops_opt
global L3CM_opt
global etime_opt
global GFLOPS_opt
global traffic_opt
global arithmetic_int_opt

# Machine Info
L3LineSize = args["l3l"] # LL cache line size
bmax = args["bm"] # maximum bandwidth;
peak_fp_performance = args["pp"] # GFLOPS/s
ridge_point = peak_fp_performance/bmax
sample_y = round(2*bmax, 2)

Flops_opt = args["fopt"]
L3CM_opt = args["l3opt"]
etime_opt = args["etopt"]

optimized = Flops_opt and L3CM_opt and etime_opt

if optimized:
    Gflops_opt = Flops_opt/(10**9)
    GFLOPS_opt = Gflops_opt/etime_opt
    traffic_opt = (L3CM_opt*L3LineSize)/(1024**3)
    arithmetic_int_opt = Gflops_opt/traffic_opt # x coordinate

# Verbose Settings
# useful for debugging
verbose_machine = args["vm"]
verbose_kernel = args["vk"]
out_file = args["out_file"]

# Some other stuffs
traffic = (L3CM*L3LineSize)/(1024**3)
arithmetic_int = Gflops/traffic

def printline():
    print("--------------------------------------------------")

def printKernel(your_kernel_name, Flops, L3CM, etime, Gflops, GFLOPS, traffic, arithmetic_int):
    printline()
    print(your_kernel_name + " - Roofline Model (RLM)")
    printline()
    print("Floating Point Operations (Flops) = " + str(Flops) + ";")
    print("L3 Cache Misses = " + str(L3CM) + ";")
    print("Execution time = " + str(etime) + ";")
    print("Gflops = " + str(Gflops) + ";")
    print("GFLOPS = " + str(GFLOPS) + ";")
    print("Traffic = " + str(traffic) + " GB;")
    print("Arithmetic Intensity (AI) = " + str(arithmetic_int) + " Gflops/GB.")
    printline()
    print("Your kernel: " + your_kernel_name+";")
    print("is: " + (arithmetic_int > ridge_point and "COMPUTE" or "MEMORY") + " BOUNDED!")
    print("with respect to the machine: ")
    print(your_machine_name+".")
    printline()
    print(" ")
    print(" ")

printline()
print("####### Roofline Model Plotter ########")
print("#######################################")
print("##### Marco Chiarelli @    CMCC  #####")
print("## marco.chiarelli@cmcc.it 20/03/2019#")
print("#######################################")
printline()


if verbose_machine == True:
    printline()
    print(your_machine_name)
    print("Listing some machine informations...")
    printline()
    print("L3LineSize = " + str(L3LineSize) + " byte;")
    print("Peak Memory BW = " + str(bmax) + " GB/s;")
    print("Peak FP Performance = " + str(peak_fp_performance) + " Gflops/s;")
    print("Corresponding Ridge point = " + str(ridge_point) + ".")
    printline()
    print(" ")
    print(" ")

if verbose_kernel == True:
    printKernel(your_kernel_name, Flops, L3CM, etime, Gflops, GFLOPS, traffic, arithmetic_int) 

    if optimized == True:
        printKernel("Optimized "+your_kernel_name, Flops_opt, L3CM_opt, etime_opt, Gflops_opt, GFLOPS_opt, traffic_opt, arithmetic_int_opt) 

if optimized == True:
    gain_performance = ((GFLOPS_opt-GFLOPS)/GFLOPS)*100;
    gain_memory = ((arithmetic_int_opt-arithmetic_int)/arithmetic_int)*100;
    print("Percentual Performance Gain: " + str(gain_performance)+ ";")
    print("Percentual Memory Gain: " + str(gain_memory) + ";")
    print("Done a " + (gain_performance > 0 and "good" or "bad") + " job wrt FP performance;")
    print("Done a " + (gain_memory > 0 and "good" or "bad") + " job wrt memory.")
    print(" ")
    print(" ")

# Preparing a straight line fitting
xP = [ridge_point, 2]
yP = [peak_fp_performance, sample_y]

coefficients = np.polyfit(xP, yP, 1)
polynomial = np.poly1d(coefficients)
x_axis = np.linspace(0,20,100)
peak_memory_BW_plt = polynomial(x_axis)
peak_fp_performance_plt = np.ones(len(x_axis))*peak_fp_performance

# PLOT SETTINGS
logbase = 2
fig, ax = plt.subplots()
# Machine RLM plot
ax.loglog(np.where(x_axis >= ridge_point, 0, x_axis),np.where(peak_memory_BW_plt >= peak_fp_performance, 0, peak_memory_BW_plt),'-',basex=logbase,basey=logbase,linewidth=2.5)
ax.loglog(np.where(x_axis >= ridge_point, x_axis, 0),np.where(peak_fp_performance_plt > peak_memory_BW_plt, 0, peak_fp_performance_plt),'o',basex=logbase,basey=logbase,linewidth=1.8)
ax.loglog(ridge_point, peak_fp_performance,'+',basex=logbase,basey=logbase,markeredgewidth=3,markersize=10)
# Kernel(s) plot
ax.loglog(arithmetic_int, GFLOPS,'o',basex=logbase,basey=logbase)

if optimized:
    ax.loglog(arithmetic_int_opt, GFLOPS_opt,'o',basex=logbase,basey=logbase)

def ticks(y, pos):
    return r'$2^{:.0f}$'.format(np.log2(y))

#ax.xaxis.set_major_formatter(mtick.FuncFormatter(ticks))
#ax.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))

# PLOT
plt.grid(True, which="both", ls="-")
plt.xlabel("Arithmetic Intensity AI [GFlops/GB]")
plt.ylabel("Attainable Performance [GFLOPS = GFlops/s]")
plt.legend(optimized == True and ["Peak Memory BW", "Peak FP Performance", "Ridge Point", "Your Kernel", "Your Optimized Kernel"] or ["Peak Memory BW", "Peak FP Performance", "Ridge Point", "Your Kernel"], loc='best')
plt.title("Roofline Model (RLM) of: " + your_kernel_name)

#if autosave:
fig.savefig(out_file) #, dpi=96 * 10)
#else:
#    plt.show()

print("Thank you for using this program.")
print(" ")
print(" ")
