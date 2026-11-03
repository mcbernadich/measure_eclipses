from scipy.optimize import differential_evolution
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import emcee

def ms_to_phase(ms):
    return ms/ps

def phase_to_ms(phase):
    return phase * ps

def hr_to_phase(hr):
    return hr/(pb*24*60) + orbital_stamps[0]

def phase_to_hr(phase):
    return (phase - orbital_stamps[0])* pb*24*60

#Identify position of peak and mask for flux
def flux_window(model):

#    plt.plot(model)
#    plt.show()

    peak=np.argmax(model)
    mask= model > 0.0

    return peak,mask

def disperse_and_scatter_template(fourier_template,ddm,scattering_scale,subbands):
    
    dm_delays=(4.148808e6*ddm/freq_channels**2)*(nbin/ps)
    scatt_delays=(scattering_scale/(freq_channels/1000)**4)*(nbin/ps)

    fourier_freqs_multi = fourier_freqs[None, :]      # shape (1, N_freq)
    dm_delays     = dm_delays[:, None]          # shape (N_chan, 1)
    scatt_delays  = scatt_delays[:, None]       # shape (N_chan, 1)

    dispersion_filter=np.exp(-2*np.pi*1j*fourier_freqs_multi*dm_delays)
    scatter_filter=1/(1+2*np.pi*1j*fourier_freqs_multi*scatt_delays)

    smeared_fourier_template=fourier_template*dispersion_filter*scatter_filter

    return np.real(np.fft.ifft(smeared_fourier_template.reshape(-1, nchan//subbands, smeared_fourier_template.shape[1]).sum(axis=1)))

def forward_residuals(pars,nband,subintegration,fourier_template):

    (ddm,st)=(pars[0],pars[1])

    smeared_template = disperse_and_scatter_template(fourier_template,ddm,st,nband)

    residuals=[]

    for i, (schannel, tchannel) in enumerate(zip(subintegration, smeared_template)):
        amplitude=max(np.sum(schannel*tchannel)/np.sum(tchannel*tchannel),0)
        residuals.append((amplitude*tchannel-schannel)/average_error[i])

    residuals=np.array(residuals)
    residuals=residuals.flatten()

    return residuals

def forward_chi2(pars,nband,subintegration,fourier_template):
    
    residuals=forward_residuals(pars,nband,subintegration,fourier_template)

    return np.sum(residuals**2)

def forward_loge_probability(pars,subintegration,nband,fourier_template):

    if pars[1]<0:
        return -np.inf
    elif pars[0]<0:
        return -0.5*forward_chi2(pars,nband,subintegration,fourier_template)+0.5*pars[0]/0.01
    else:
        return -0.5*forward_chi2(pars,nband,subintegration,fourier_template)

def fit_forward(subbanded_data,fourier_template):

    chi2rs=[]

    dms=[]
    tss=[]

    ddms=[]
    dtss=[]

    print("")
#    ii=[27,34]
#    for i in ii:
    i=0
    for subintegration in subbanded_data:
        
#        subintegration=subbanded_data[i]
        print("Fitting",i,"out of",nsub,"subintegrations.")

        #Fit with scipy first
        par = differential_evolution(forward_chi2, x0=[0.0,0.0], args=(nsubbands,subintegration,fourier_template), bounds=[(-0.2,0.2),(0.0,1.0)])
        result = least_squares(forward_residuals, par.x, args=(nsubbands,subintegration,fourier_template), bounds=[(-0.2,0.0),(0.2,1.0)])
        chi2=np.sum(result.fun**2)
        chi2r=chi2/(nsubbands*nbin-2)
         
        chi2rs.append(chi2r)
        dms.append(result.x[0])
        tss.append(result.x[1])

        #Uncertaintise with emcee
        ndim = 2
        nwalkers = 8
        par0 = np.array(result.x)
        pos = par0 + 1e-2 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers,ndim,forward_loge_probability,args=(subintegration, nsubbands, fourier_template))
        nsteps = 300
        sampler.run_mcmc(pos, nsteps, progress=False)
        samples = sampler.get_chain(discard=50, flat=True)

        ddm_unc=np.percentile(samples[:, 0],[17,83])
        ts_unc=np.percentile(samples[:, 1],[17,83])

        ddms.append((ddm_unc[1]-ddm_unc[0])/2)

        #Check if the posterior of scattering is piling at 0 or not.
        unc=(ts_unc[1]-ts_unc[0])/2
        if unc<result.x[1]:
            dtss.append(unc)
        else:
            ts_unc=np.percentile(samples[:, 1],66)
            dtss.append(ts_unc)    

        i=i+1

    return np.array(chi2rs),np.array(dms),np.array(ddms),np.array(tss),np.array(dtss)

def dedisperse_and_unscatter_subintegration(subintegration,ddm,sts,subbands):
    #Un-scattering is implemented via time-domain shifting, as pure un-scattering would mostly just un-scatter noise.
    #I have tested and the peak of the profile shifts a length equal to the usncattered profile, so it is viable.

    dm_delays=(4.148808e6*ddm/freq_channels**2)*(nbin/ps)
    scatt_delays=(sts/(freq_channels/1000)**4)*(nbin/ps)

    fourier_subintegration=np.fft.fft(subintegration)

    fourier_freqs_multi = fourier_freqs[None, :]      # shape (1, N_freq)
    dm_delays     = dm_delays[:, None]          # shape (N_chan, 1)
    scatt_delays  = scatt_delays[:, None]       # shape (N_chan, 1)

    dispersion_filter=np.exp(-2*np.pi*1j*fourier_freqs_multi*(-dm_delays))  #Just do it in the opposite direction
    scatter_filter=np.exp(-2*np.pi*1j*fourier_freqs_multi*(-scatt_delays))  #Just do it in the opposite direction

    corrected_fourier_subintegration=fourier_subintegration*dispersion_filter*scatter_filter
    corrected_fourier_subintegration = corrected_fourier_subintegration.reshape(subbands,-1,corrected_fourier_subintegration.shape[1]).sum(axis=1)
    summed_subintegration=np.real(np.fft.ifft(corrected_fourier_subintegration))

    return summed_subintegration

def compute_flux(data,fouriter_template,ddms,dddms,sts,dsts):

    fluxes=[]
    flux_errors=[]

    print("")
    print("Computing fluxes")
    i=0
    for subintegration in data:

        #Integrate 
        summed_subintegration=dedisperse_and_unscatter_subintegration(subintegration,ddms[i],sts[i],nsubbands)
        smeared_fourier_template=disperse_and_scatter_template(fourier_template,ddms[i],sts[i],nchan)
        summed_template=dedisperse_and_unscatter_subintegration(smeared_fourier_template,ddms[i],sts[i],nsubbands)
        
        for k in range(0,8):
            amplitude=max(np.sum(summed_subintegration[k]*summed_template[k])/np.sum(summed_template[k]*summed_template[k]),0)
            summed_template[k]=amplitude*summed_template[k]

        summed_subintegration=np.sum(summed_subintegration,axis=0)
        summed_template=np.sum(summed_template,axis=0)

        #Compute flux from nominal values
        amplitude=np.sum(summed_subintegration*summed_template)/np.sum(summed_template*summed_template)
        scaled_template=amplitude*summed_template
        flux=np.sum(scaled_template)
        fluxes.append(flux)

        #Compute error on flux from amplitude uncertainy
        error1=np.median(abs(summed_subintegration))
        error2=np.std(summed_subintegration[scaled_template<error1])
        amplitude_error=error2/np.sqrt(np.sum(summed_template*summed_template))
        flux_error_from_amplitude=amplitude_error*np.sum(summed_template)

        flux_errors.append(flux_error_from_amplitude)  #Order of magnitude larger than the DM and scattering contributions

        chi2=np.sum(((scaled_template-summed_subintegration)/error2)**2)
#        print(chi2,chi2/(nbin-1),flux,flux_error_from_amplitude)

        #Chech if the measurement is actually emaningful.
        if np.max(scaled_template)<error2:
            fluxes[i]=0
            flux_errors[i]=abs(np.sum(summed_subintegration))

        i=i+1

    fluxes=np.array(fluxes)
    flux_errors=np.array(flux_errors)
    
    fluxes[fluxes<0]=0

    return fluxes,flux_errors

parser=argparse.ArgumentParser(description="Take in an ascii file with the phase-time info, and plot the eclipse over time.")
parser.add_argument("--name",help="Title of the plot.")
parser.add_argument("-d","--data",help="Ascii file with data. Could be any set amount of channels.")
parser.add_argument("-s","--model",help="Analytic template.")
parser.add_argument("-t","--time",help="List of ToAs, tempo2 format.")
parser.add_argument("-n","--nsubbands",type=int,help="Number of sub-bands to implement.")
parser.add_argument("--color_mask",help="Orbital phase mask for color scaling (do not conside outliers).")
parser.add_argument("--ps",type=float,help="Spin period in ms.")
parser.add_argument("--pb",type=float,help="Orbital period in days.")
parser.add_argument("--ta",type=float,help="Time of ascending node (MJD).")
parser.add_argument("-l","--label",help="Title of plot")
args = parser.parse_args()

ps=args.ps
pb=args.pb
ta=args.ta

#Load all the data sets
print("")
print("Loading "+args.data+" ...")
data=np.loadtxt(args.data,delimiter=" ", skiprows=1).T
standard_model=np.loadtxt(args.model,delimiter=" ", skiprows=1).T
standard_model=standard_model[3]
standard_model[standard_model<1e-11]=0.0 #Make sure it is trully outside of pofile
time=np.loadtxt(args.time,dtype="str").T

#Identify position of peak and mask for flux (pulse_bins)
(peak,pulse_bins)=flux_window(standard_model)

#Read the header of the data file to get some useful info
read_header=open(args.data,"r")
header=read_header.readline()
print(header.split("\n")[0])
psr=header.split("File: ")[1].split("_")[0]
ephemeris=header.split("File: ")[1].split("_")[1].split(".")[0]
date=ephemeris[0:10]
nsub=int(header.split("Nsub:")[1].split("Nch:")[0])-1     #Always remove the last one
nbin=int(header.split("Nbin:")[1].split("RMS")[0])
nchan=int(header.split("Nch:")[1].split("Npol")[0])
freq=float(header.split("freq:")[1].split("bw:")[0])
bw=float(header.split("bw:")[1].split("dm:")[0])
should_peak=int(nbin*0.25)
read_header.close()

#Create array with the frequency channels
freq_channels=np.linspace(freq-bw/2+bw/nchan/2,freq+bw/2-bw/nchan/2,nchan)

#Create arrays with the orbital phase (derived from time) and the spin phase
orbital_stamps=(time[2,:-1].astype("float")-ta)/pb
start=orbital_stamps[0]
end=orbital_stamps[-1]
orbital_phase_array=np.linspace(start,end,nsub)
orbital_phase_array=orbital_phase_array-int(orbital_phase_array[0])
orbital_stamps=orbital_stamps-int(orbital_stamps[0])
dorb=orbital_phase_array[1]-orbital_phase_array[0]
spin_phase_array=(data[2,0:nbin]+0.5)/nbin
(orbit_matrix,spin_matrix)=np.meshgrid(orbital_phase_array,spin_phase_array)

#Create a 3D array with data across subintegration, channels, and bins.
intensity_matrix=data[3,:-nbin*nchan].reshape(nsub,nchan,nbin)

#Integrate the matrix in time (for color scale and plotting later on)
integrated_intensity_matrix=np.sum(intensity_matrix,axis=1)

#Subb-band the data
nsubbands=args.nsubbands
n1, n2, n3 = intensity_matrix[:,8:-8,:].shape
subbanded_intensity_matrix = intensity_matrix[:,8:-8,:].reshape(n1, nsubbands, n2//nsubbands, n3).sum(axis=2)

#Compute average error on the sub-banded data (off-pulse noise), and flux error
average_error=np.std(subbanded_intensity_matrix[:,:,~pulse_bins],axis=(0,2))
flux_error_per_band=np.sqrt(np.sum(pulse_bins))*average_error
total_flux_error=np.sqrt(np.sum(flux_error_per_band**2))

#Create a channel mask for the template from the intensity_matrix
zap_mask = []
i=0
for row in intensity_matrix[0, :, :]:
    if np.all(row == 0) or i==0 or i==(nbin-1):    #This is just for 0-ing a missing unzap line.
        zap_mask.append(0)
    else:
        zap_mask.append(1)
    i=i+1
zap_mask = np.array(zap_mask)

#Expand the template matrix to the same frequency resolution as the template
template_matrix=zap_mask[:, np.newaxis]*np.tile(standard_model, (nchan, 1))
template_matrix=template_matrix[8:-8,:]

#Trimming for S/N purposes
freq_channels=freq_channels[8:-8]
nchan=nchan-16

#Move the template to the Fourier domain
fourier_template=np.fft.fft(template_matrix)
fourier_freqs=np.fft.fftfreq(nbin)

#The tempalte plotting aprt goes here if you want it.

#Fit DM and scattering time-scales at each subintegration (or read them from file)
try:
    measurements=np.loadtxt(args.data+".measurements_"+str(nsubbands)+"bands",delimiter=",").T
    chi2r_values=measurements[0]
    dm_values=measurements[1]
    dm_errors=measurements[2]
    ts_values=measurements[3]
    ts_errors=measurements[4]
except:
#if 1!=0:
    chi2r_values,dm_values,dm_errors,ts_values,ts_errors=fit_forward(subbanded_intensity_matrix,fourier_template)
    with open(args.data+".measurements_"+str(nsubbands)+"bands","w") as write_file:
        for (chi2r,dm,dm_error,ts,ts_error) in zip(chi2r_values,dm_values,dm_errors,ts_values,ts_errors):
            write_file.write(f"{chi2r}, {dm}, {dm_error}, {ts}, {ts_error}\n")
    write_file.close()

#Fit fluxes:
fluxes,flux_errors=compute_flux(intensity_matrix[:,8:-8,:],fourier_template,dm_values,dm_errors,ts_values,ts_errors)
window=int(13)
kernel=np.ones(window)/window
smoothed_flux_curve=np.convolve(fluxes,kernel,mode="same")
flux_errors=flux_errors/np.max(smoothed_flux_curve)
fluxes=fluxes/np.max(smoothed_flux_curve)
smoothed_flux_curve=smoothed_flux_curve/np.max(smoothed_flux_curve)

#Also smoothen the DM and scattering timescale

#Set a mask of elements to plot
plot_mask= ( ((fluxes-3*flux_errors)>0) & (ts_errors<0.5) & (dm_errors<0.5) & ( (ts_values + ts_errors) < 1.0 ) )

with open(args.data+".eclipseTF_"+str(nsubbands)+"bands", "w") as f:
    for tstamp, value, boolean in zip(time[2,:-1], orbital_stamps, ~plot_mask):
        f.write(f"{tstamp} {value} {int(boolean)}\n")

#Plot
#fig, axes = plt.subplots(nrows=5,ncols=1,figsize=(9*(np.max(orbital_phase_array)/1.0), 8))
fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(9, 9),gridspec_kw={'height_ratios': [1.3, 1, 1, 1]})
fig.subplots_adjust(hspace = 0.05,top=0.894, bottom=0.08)
#fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(9, 6),gridspec_kw={'height_ratios': [1.3, 1, 1]})
#fig.subplots_adjust(hspace = 0.05,top=0.85, bottom=0.08)

#Just an optional masking for the color
if args.color_mask:
    min_phase=float(args.phase_mask.split(":")[0])
    max_phase=float(args.phase_mask.split(":")[1])
    vmin = np.percentile(integrated_intensity_matrix[ (orbital_phase_array < min_phase) | (orbital_phase_array > max_phase) ], 0.5)  # 5th percentile
    vmax = np.percentile(integrated_intensity_matrix[ (orbital_phase_array < min_phase) | (orbital_phase_array > max_phase) ], 99.9)  # 95th percentile
else:
    vmin = np.percentile(integrated_intensity_matrix, 0.5)  # 5th percentile
    vmax = np.percentile(integrated_intensity_matrix, 99.9)  # 95th percentile
norm = PowerNorm(gamma=0.8,vmin=vmin,vmax=vmax)

#Make the plots.
axes[0].pcolormesh(orbital_phase_array, spin_phase_array, np.roll(integrated_intensity_matrix,-(peak-should_peak),axis=1).T, cmap='afmhot', norm=norm,shading='auto')
axes[1].errorbar(orbital_stamps[plot_mask], fluxes[plot_mask], yerr=flux_errors[plot_mask], fmt='.',zorder=10)
#axes[1].plot(orbital_stamps, smoothed_flux_curve,zorder=15)
axes[2].errorbar(orbital_stamps[plot_mask], dm_values[plot_mask], yerr=dm_errors[plot_mask], fmt='.')
axes[3].errorbar(orbital_stamps[plot_mask], ts_values[plot_mask], yerr=ts_errors[plot_mask], fmt='.')
#axes[4].scatter(orbital_stamps, chi2r_values, marker='.')

#Set axis labels.
if args.name:
    fig.suptitle(args.name+" ("+date+", "+str(round(freq))+" MHz)",fontsize=18)
else:
    fig.suptitle(ephemeris+", "+str(round(freq))+" MHz",fontsize=18)
axes[0].set_ylabel("Spin phase",fontsize=16)
axes[1].set_ylabel("Flux",fontsize=16)
axes[2].set_ylabel(r"$\Delta$DM (pc cm$^{-3}$)",fontsize=16)
axes[3].set_ylabel(r"$\tau_\mathrm{s, 1\,GHz}$ (ms)",fontsize=16)
axes[3].set_xlabel("Orbital phase, $\phi$ (mean latitude)",fontsize=16)
#axes[4].set_ylabel(r"$\chi^2_\mathrm{r}$",fontsize=13)
#axes[4].set_xlabel(r"$\phi_\mathrm{b}$ (mean anomaly)",fontsize=13)

#Set secondary axes
#ax_delays=axes[0].secondary_yaxis('right', functions=(phase_to_ms, ms_to_phase))
#ax_delays.set_ylabel(r"$\phi_\mathrm{s}$ (ms)",fontsize=13)
ax_hours=axes[0].secondary_xaxis('top', functions=(phase_to_hr, hr_to_phase))
ax_hours.set_xlabel("Observation time (minutes)",fontsize=16)
ax_hours1=axes[1].secondary_xaxis('top', functions=(phase_to_hr, hr_to_phase))
ax_hours1.tick_params(axis='x',direction='in',labeltop=False)
ax_hours2=axes[2].secondary_xaxis('top', functions=(phase_to_hr, hr_to_phase))
ax_hours2.tick_params(axis='x',direction='in',labeltop=False)
ax_hours3=axes[3].secondary_xaxis('top', functions=(phase_to_hr, hr_to_phase))
ax_hours3.tick_params(axis='x',direction='in',labeltop=False)
#ax_hours4=axes[4].secondary_xaxis('top', functions=(phase_to_hr, hr_to_phase))
#ax_hours4.tick_params(axis='x',direction='in',labeltop=False)

#Add in superior conjunction.
axes[1].vlines([0.25,1.25,2.25],-10,10,colors="r",linestyles="dashed",label="Superior conjunction")
axes[2].vlines([0.25,1.25,2.25],-10,10,colors="r",linestyles="dashed")
axes[3].vlines([0.25,1.25,2.25],-10,10,colors="r",linestyles="dashed")
#axes[4].vlines([0.25,1.25,2.25],-10,10,colors="r",linestyles="dashed")

#Set axis limits.
axes[0].set_xlim(orbital_phase_array[0]-dorb/2,orbital_phase_array[-2]+dorb/2)
axes[1].set_xlim(orbital_phase_array[0]-dorb/2,orbital_phase_array[-2]+dorb/2)
axes[2].set_xlim(orbital_phase_array[0]-dorb/2,orbital_phase_array[-2]+dorb/2)
axes[3].set_xlim(orbital_phase_array[0]-dorb/2,orbital_phase_array[-2]+dorb/2)
#axes[4].set_xlim(orbital_phase_array[0]-dorb/2,orbital_phase_array[-2]+dorb/2)

axes[0].set_ylim(0,1)
axes[1].set_ylim(-0.01,1.35)
ddm_std=np.std(dm_values)
axes[2].set_ylim(-0.04,2.1*ddm_std)
ts_std=np.std(ts_values)
axes[3].set_ylim(-0.15,2.5*ts_std)
#axes[4].set_ylim(0.6,2.2)

#Set legend
axes[1].legend(fontsize=12)

#Set tick labels
ax_hours.tick_params(axis='both',labelsize=15)
axes[0].tick_params(axis='both',labelsize=15)
axes[0].tick_params(bottom=False,labelbottom=False)
axes[0].xaxis.labelpad = 8
axes[1].tick_params(axis='both',labelsize=15)
axes[1].tick_params(axis='x',direction='in',labelbottom=False)
axes[1].xaxis.labelpad = 8
axes[2].tick_params(axis='both',labelsize=15)
axes[2].tick_params(axis='x',direction='in',labelbottom=False)
axes[2].xaxis.labelpad = 8
axes[3].tick_params(axis='both',labelsize=15)
axes[3].xaxis.set_major_formatter( FuncFormatter(lambda value, pos: f"{value % 1.0:g}"))

# Save as PDF
fig.savefig(psr+"_"+ephemeris+".pdf", bbox_inches="tight")

# Save as PNG
fig.savefig(psr+"_"+ephemeris+".png", dpi=400, bbox_inches="tight")

# Show
#plt.show()