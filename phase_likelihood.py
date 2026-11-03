from astropy.timeseries import LombScargle
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import argparse
import emcee
import sys

def lnL_uniform(pars,phases,eclipsed):

    (pu)=(pars[0])

    if not (0 <= pu < 1):
        return -np.inf

    p = pu + 0*phases
    p[eclipsed==0] = 1 - p[eclipsed==0]

    lnL=np.sum(np.log(p))

    return lnL

def lnL_cos_fixed(pars,phases,eclipsed):

    (pu,C)=(pars[0],pars[1])

    if not (0 <= pu < 1 and C >= 0):
        return -np.inf

    p = pu + C * np.cos(2*np.pi*(phases-0.25))

    p = np.minimum(p, 1.0)
    p = np.maximum(p, 0.0)
    p[eclipsed==0] = 1 - p[eclipsed==0]

    lnL=np.sum(np.log(p))

    return lnL

def lnL_cos_free(pars,phases,eclipsed):

    (pu,C,peak)=(pars[0],pars[1],pars[2])

    if not (0 <= pu < 1 and C >= 0 and -0.25 <= peak < 0.75):
        return -np.inf

    p = pu + C * np.cos(2*np.pi*(phases-peak))
    p = np.minimum(p, 1.0)
    p = np.maximum(p, 0.0)
    p[eclipsed==0] = 1 - p[eclipsed==0]

    lnL=np.sum(np.log(p))

    return lnL

def lnL_vonMisses(pars,sigma,phases,eclipsed):

    (pu,H)=(pars[0],pars[1])

    if not (0 <= pu < 1 and H >= 0):
        return -np.inf

    p = pu + H * np.exp((np.cos(2*np.pi*(phases-0.25))-1)/(4*np.pi**2*sigma**2))
    p = np.minimum(p, 1.0)
    p[eclipsed==0] = 1 - p[eclipsed==0]

    lnL=np.sum(np.log(p))

    return lnL

parser=argparse.ArgumentParser(description="Compute the likelihood describing the periodicity of ecplipses.")
parser.add_argument("-p","--phases",help="List of files with 'orbital_phase eclipse' (float 0/1), separated by ','")
parser.add_argument("-c","--cos",action="store_true",help="Try a uniform + cos distribution.")
parser.add_argument("-v","--vonMisses",type=float,help="Try a uniform + von Misses distribution. State a float-valued width of the eclipse window")
args = parser.parse_args()

data_files=args.phases.split(",")

tstamps=np.array([]).astype(float)
phases=np.array([]).astype(float)
eclipsed=np.array([]).astype(int)

for file in data_files:
    data=np.loadtxt(file,dtype=str,delimiter=" ").T
    tstamps=np.concatenate((tstamps,data[0].astype(float)))
    phases=np.concatenate((phases,data[1].astype(float)))
    eclipsed=np.concatenate((eclipsed,data[2].astype(int)))

tstamps=tstamps-np.min(tstamps)
phases=np.mod(phases,1.0)

print("")
print("Computing a Lomb-Scargle diagram")
ls=LombScargle(24*tstamps,eclipsed)
(frequencies,power)=ls.autopower(minimum_frequency=1/(24*30),maximum_frequency=1/(24*0.01),samples_per_peak=1)

plt.plot(frequencies,power)
plt.xlabel(r"Frequency (hr$^{-1}$)",fontsize=16)
plt.ylabel(r"Lomb-Scargle power",fontsize=16)
plt.yticks([])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0,100/24)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("Lomb-Scargle.png")
plt.show()

print(phases.size,np.sum(eclipsed))

plt.hist(phases,bins=50,color="c",label="All sub-integrations")
plt.hist(phases[eclipsed==1],bins=50,color="k",label="Eclipsed sub-integrations")
plt.xlabel(r"Orbital phase (mean latitude)",fontsize=16)
plt.ylabel(r"Samples",fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0,1)
plt.ylim(top=80)
plt.tight_layout()
plt.savefig("samples.pdf")
plt.close()

if args.cos:

    print("")
    print("Fitting uniform + cosine peaked at phase 0.25")

    ndim = 2
    nwalkers = 16
    par0 = np.array([0.5,0.25])
    pos = par0 + 0.1 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL_cos_fixed,args=(phases,eclipsed))
    nsteps = 1000
    sampler.run_mcmc(pos, nsteps, progress=False)
    samples = sampler.get_chain(discard=100, flat=True)
    lnL = sampler.get_log_prob(discard=100, flat=True)    
    first_guess = samples[np.argmax(lnL)]
    results = minimize(lambda x: -lnL_cos_fixed(x, phases, eclipsed), first_guess, method='Nelder-Mead')

    pu_c_fixed=results.x[0]
    C_fixed=results.x[1]
    samples_c_fixed=samples

    pu_edges=np.percentile(samples_c_fixed[:, 0],[17,83])
    pu_uncertainty_fixed=(pu_edges[1]-pu_edges[0])/2
    print(pu_c_fixed,pu_uncertainty_fixed)

    C_edges=np.percentile(samples_c_fixed[:, 1],[17,83])
    C_uncertainty_fixed=(C_edges[1]-C_edges[0])/2
    print(C_fixed,C_uncertainty_fixed)
    print(lnL_cos_fixed(results.x,phases,eclipsed))

    phases_array=np.arange(0,1.001,0.001)
    p_array_fixed= pu_c_fixed + C_fixed * np.cos(2*np.pi*(phases_array-0.25))
    phases_array_coarse=np.arange(0,1.01,0.01)
    p_array_array_fixed= samples_c_fixed[:,0,np.newaxis] + samples_c_fixed[:,1,np.newaxis] * np.cos(2*np.pi*(phases_array_coarse-0.25))
    plot_name="eclipsed_fraction+sinusoidal.png"

    plt.plot(samples_c_fixed[:, 0], samples_c_fixed[:, 1],".g",alpha=0.05)
    plt.plot(results.x[0],results.x[1],"*k",markersize=20,label="Best-fit values")
    plt.plot([],[],".g",label="emcee samples")
    plt.xlabel(r"Uniform probability amplitude, $p_\mathrm{u}$",fontsize=16)
    plt.ylabel(r"Sinusoid amplitude, $C$",fontsize=16)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("uniform+sinusoid_posterior.pdf")
    plt.close()

    print("")
    print("Fitting uniform + cosine peaked at free phase")

    ndim = 3
    nwalkers = 16
    par0 = np.array([0.5,0.25,0.25])
    pos = par0 + 0.1 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL_cos_free,args=(phases,eclipsed))
    nsteps = 2000
    sampler.run_mcmc(pos, nsteps, progress=False)
    samples = sampler.get_chain(discard=100, flat=True)
    lnL = sampler.get_log_prob(discard=100, flat=True)
    first_guess = samples[np.argmax(lnL)]
    results = minimize(lambda x: -lnL_cos_free(x, phases, eclipsed), first_guess, method='Nelder-Mead')

    pu_c_free=results.x[0]
    C_free=results.x[1]
    peak=results.x[2]
    samples_c_free=samples

    pu_edges=np.percentile(samples_c_free[:, 0],[17,83])
    pu_uncertainty_free=(pu_edges[1]-pu_edges[0])/2
    print(pu_c_free,pu_uncertainty_free)

    C_edges=np.percentile(samples_c_free[:, 1],[17,83])
    C_uncertainty_free=(C_edges[1]-C_edges[0])/2
    print(C_free,C_uncertainty_free)

    peak_edges=np.percentile(samples_c_free[:, 2],[17,83])
    peak_uncertainty=(peak_edges[1]-peak_edges[0])/2
    print(np.mod(peak,1.0),peak_uncertainty)
    print(lnL_cos_free(results.x,phases,eclipsed))

#    phases_array=np.arange(0,1.001,0.001)
    p_array_free= pu_c_free + C_free * np.cos(2*np.pi*(phases_array-peak))
#    phases_array_coarse=np.arange(0,1.01,0.01)
    p_array_array_free= samples_c_free[:,0,np.newaxis] + samples_c_free[:,1,np.newaxis] * np.cos(2*np.pi*(phases_array_coarse-samples_c_free[:,2,np.newaxis]))
    plot_name="eclipsed_fraction+sinusoidal.png"

    plt.plot(samples_c_free[:, 2], samples_c_free[:, 1],".g",alpha=0.05)
    plt.plot(results.x[2],results.x[1],"*k",markersize=20,label="Best-fit values")
    plt.plot([],[],".g",label="emcee samples")
    plt.xlabel(r"Peak eclipse phase, $\phi_0$",fontsize=16)
    plt.ylabel(r"Sinusoid amplitude, $C$",fontsize=16)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("uniform+free_sinusoid_posterior.pdf")
    plt.close()

elif args.vonMisses:

    sigma=args.vonMisses

    print("")
    print("Fitting uniform + von Misses centered at phase 0.25 with width "+str(sigma))

    ndim = 2
    nwalkers = 16
    par0 = np.array([0.5,0.5])
    pos = par0 + 0.1 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL_vonMisses,args=(sigma,phases,eclipsed))
    nsteps = 1000
    sampler.run_mcmc(pos, nsteps, progress=False)
    samples = sampler.get_chain(discard=100, flat=True)
    lnL = sampler.get_log_prob(discard=100, flat=True)    
    first_guess = samples[np.argmax(lnL)]
    results = minimize(lambda x: -lnL_vonMisses(x, sigma, phases, eclipsed), first_guess, method='Nelder-Mead')

    phases_array=np.arange(0,1.001,0.001)
    p_array= results.x[0] + results.x[1] * np.exp((np.cos(2*np.pi*(phases_array-0.25))-1)/(4*np.pi**2*sigma**2))
    plot_name="eclipsed_fraction+vonMisses_s"+str(int(100*sigma))+".png"

    plt.plot(samples[:, 0], samples[:, 1],".c",alpha=0.05)
    plt.plot(results.x[0],results.x[1],"*k",markersize=20,label="Best-fit values")
    plt.plot([],[],".c",label="emcee samples")
    plt.xlabel(r"Uniform probability amplitude, $p_\mathrm{u}$",fontsize=16)
    plt.ylabel(r"von Misses prob. amplitude, $H$",fontsize=16)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("uniform+vonMisses_posterior_s"+str(int(100*sigma))+".pdf")
    plt.close()

print("")
print("Fitting uniform only")

ndim = 1
nwalkers = 16
par0 = np.array([0.5])
pos = par0 + 0.1 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL_uniform,args=(phases,eclipsed))
nsteps = 1000
sampler.run_mcmc(pos, nsteps, progress=False)
samples = sampler.get_chain(discard=100, flat=True)
lnL = sampler.get_log_prob(discard=100, flat=True)    
first_guess = samples[np.argmax(lnL)]
results = minimize(lambda x: -lnL_uniform(x, phases, eclipsed), first_guess, method='Nelder-Mead')

pu=results.x[0]
pu_edges=np.percentile(samples[:, 0],[17,83])
pu_uncertainty=(pu_edges[1]-pu_edges[0])/2
samples_u=samples
print(pu,pu_uncertainty)
print(lnL_uniform(results.x,phases,eclipsed))

plot_name="eclipsed_fraction.pdf"

bins = np.linspace(0, 1, 51)
mean_eclipsed, edges = np.histogram(
    phases,
    bins=bins,
    weights=eclipsed
)
counts, _ = np.histogram(phases, bins=bins)
mean_eclipsed = mean_eclipsed / counts
error=np.sqrt(mean_eclipsed*(1-mean_eclipsed)/counts)

bin_centers = (edges[:-1] + edges[1:]) / 2

plt.figure(figsize=(6.4, 3.84))

#plt.plot(phases_array,p_array_fixed,"k-",label="Uniform + sinusoid",zorder=10)
#plt.plot(phases_array,p_array_free,"k--",label="Uniform + fixed sinusoid",zorder=10)
plt.hlines(pu,0,1,color="k",linestyle="--",label="Mean eclipsed fraction (34%)",zorder=10)
#plt.hlines(samples_u[::20,0],0,1,color="g",linestyle="-",alpha=0.01,zorder=9)
#for array in p_array_array_fixed[::20]:
#    plt.plot(phases_array_coarse,array,"g-",alpha=0.01,zorder=8)
#for array in p_array_array_free[::20]:
#    plt.plot(phases_array_coarse,array,"r-",alpha=0.01,zorder=8)
plt.errorbar(bin_centers, mean_eclipsed,yerr=error,fmt="bo-",label="Samples",zorder=7)
plt.xlabel(r"Orbital phase (mean latitude)", fontsize=16)
plt.ylabel("Eclipsed fraction", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlim(0,1)
plt.ylim(0,0.8)
plt.tight_layout()
plt.savefig(plot_name)
plt.close()