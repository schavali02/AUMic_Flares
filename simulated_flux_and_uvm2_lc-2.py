# Graphing Simulated Photometric Flux Light Curve
# FIGURE OUT WHY THE FLUX IS AN ORDER OF MAGNITUDE HIGHER, DO A CALCULATION CHECK
# CONVERT Y ERROR FOR SIMULATED FLUX TO COUNTS/S
# GRAPH THE SIMULATED PHOTOMETRIC FLUX AND THE UVM2 FLUX ON THE SAME PLOT
# IF YOU ARE ABLE TO DO THESE THEN MEASURE THE FLARE PROPERTIES LIKE DURATION, TOTAL FLUX, ETC.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.time import Time
from matplotlib.lines import Line2D

plt.ion()

# getting wavelength and effective area from swift website file
fname = "/Users/shashankchavali/Downloads/Swift_UVOT.UVM2.dat"
x = pd.read_table(fname, header=1, delim_whitespace=True,
                  names=['wavelength', 'effective area'])
wave = np.array(x['wavelength'])
area = np.array(x['effective area'])
h = 6.62e-27  # cm2 g s-1
c = 3e10  # cm/s

plt.figure()

#### ALLISON'S ADDITION #################
time_array = np.array([])
isot_array = np.array([])
exposure_time_array = np.array([])
simulated_photometric_flux_array = np.array([])
simulated_photometric_flux_error_array = np.array([])
filename_array = np.array([], dtype=str)
ra_array = np.array([])
dec_array = np.array([])
roll_angle_array = np.array([])

#########################################

# Simulated Flux
rootdir = "/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_AY/"
for dirName, subdirList, fileList in os.walk(rootdir):
    for file in fileList:
        if file != ".DS_Store":
            # getting the wavelength and flux from all the grism files
            # doing conversion on grism flux density to get the simulated photometric flux
            grism_name = rootdir+"/"+file
            grism_plot = pd.read_table(
                grism_name, header=1, delim_whitespace=True, names=['wave', 'flux', 'error'])
            g_wave = np.array(grism_plot['wave'])  # angstrom
            mask = g_wave > 2200  # include only wavelengths longer than 2200!
            g_flux = np.array(grism_plot['flux'])  # erg cm-2 s-1 A-1
            effective_area_array_interp = np.interp(g_wave, wave, area)  # cm^2

            photon_energy = (h * c) / (g_wave / 1e8)  # erg

            g_photon_flux = g_flux / photon_energy  # 1/(cm^2*s*A)

            simulated_photometric_flux = np.trapz(
                effective_area_array_interp[mask] * g_photon_flux[mask], g_wave[mask])  # 1/s - AY added mask here

            # grism_uvm2_flux_array = g_flux * effective_area_array_interp (erg/s/A)
            # below is the simulated photometric flux from the email units (erg/s)
            # simulated_photometric_flux = np.trapz(grism_uvm2_flux_array, g_wave)

            # getting the y errors from their files
            flux_error_file = "/Users/shashankchavali/projects/Python/ResearchProject/overlayitems/" + \
                file[:13]+".txt"
            flux_error_read = pd.read_table(flux_error_file, header=5, delim_whitespace=True, names=[
                'wave(A)', 'weighted flux(erg cm-2 s-1 A-1)', 'variance weighted flux', 'flux(erg cm-2 s-1 A-1)', 'flux error (deviations from mean)', 'flux error (mean noise)', 'number of data summed', 'sector'])
            # units: erg cm-2 s-1 A-1
            flux_error = np.mean(
                np.array(flux_error_read['flux error (mean noise)']))
            ###### NEW ERROR BAR CODE #######
            delta_lambda = g_wave[1] - g_wave[0]
            photon_flux_error = flux_error / photon_energy  # 1/(cm^2*s*A)
            simulated_flux_error = np.sqrt(np.sum(
                (effective_area_array_interp[mask] * photon_flux_error[mask])**2)) * delta_lambda
            #################################
            x_error_file = "/Users/shashankchavali/projects/Python/ResearchProject/files_for_Allison/sw" + \
                file[:11]+"ugu_" + file[12:13] + "_detect.fits"
            if os.path.exists(x_error_file) == True or x_error_file == '/Users/shashankchavali/projects/Python/ResearchProject/files_for_Allison/sw00095500034ugu_1_detect.fits':
                hdu = fits.open(x_error_file)
                # getting the x errors and x value and plotting the error bar
                x_error = hdu[1].header['TELAPSE'] / 2
                start_time = Time(hdu[1].header["DATE-OBS"],
                                  format="isot", scale="utc")
                end_time = Time(hdu[1].header["DATE-END"],
                                format="isot", scale="utc")
                t_mid_jd = np.mean([start_time.jd, end_time.jd])
                mean_time = Time(t_mid_jd, format="jd")
                x = mean_time.jd - 2458944.954386574
                x = x * 24 * 3600
                plt.errorbar(x, simulated_photometric_flux,
                             yerr=simulated_flux_error, xerr=x_error, fmt="o", c="red")

                ##### ALLISON'S ADDITION####
                time_array = np.append(time_array, mean_time.jd)
                isot_array = np.append(isot_array, mean_time.isot)
                exposure_time_array = np.append(
                    exposure_time_array, hdu[1].header['TELAPSE'])
                simulated_photometric_flux_array = np.append(
                    simulated_photometric_flux_array, simulated_photometric_flux)
                simulated_photometric_flux_error_array = np.append(
                    simulated_photometric_flux_error_array, simulated_flux_error)
                filename_array = np.append(filename_array, file)
                ra_array = np.append(ra_array, hdu[1].header['RA_PNT'])
                dec_array = np.append(dec_array, hdu[1].header['DEC_PNT'])
                roll_angle_array = np.append(
                    roll_angle_array, hdu[1].header['PA_PNT'])

                ############################

            else:
                print(x_error_file + ' path does not exist')


##### ALLISON'S ADDITION ##############
# np.savetxt('/Users/aayoungb/Data/Swift/AUMic_grism_flares/Shashank_code_UVM2_synthetic_photometry/AUMic_UVM2_synthetic_photometry.txt',
#            np.transpose(np.array([time_array,exposure_time_array,simulated_photometric_flux_array,simulated_photometric_flux_error_array,
#                filename_array])),
#            header='time(jd),exptime(s),flux(counts/s),error(counts/s),filename',delimiter=',',fmt='%s')

df = pd.DataFrame(data={'Time (MJD)': time_array, 'Time (ISOT)': isot_array, 'RA (deg)': ra_array,
                        'Dec (deg)': dec_array, 'Roll Angle (deg)': roll_angle_array, 'exptime (s)': exposure_time_array,
                        'Flux (counts/s)': simulated_photometric_flux_array, 'Error (counts/s)': simulated_photometric_flux_error_array,
                        'filename': filename_array})
df['Type'] = 'UV grism'
df.to_csv('/Users/shashankchavali/projects/Python/ResearchProject/AUMic_UVM2_synthetic_photometry.txt')
########################################


# UVM2 Flux
rootdir = "/Users/shashankchavali/Downloads/uvotsource_output"
# iterating through the reduced uvm2 files


#### ALLISON'S ADDITION #################
time_array = np.array([])
isot_array = np.array([])
exposure_time_array = np.array([])
photometric_flux_array = np.array([])
photometric_flux_error_array = np.array([])
filename_array = np.array([], dtype=str)
ra_array = np.array([])
dec_array = np.array([])
roll_angle_array = np.array([])
#########################################


for dirName, subdirList, fileList in os.walk(rootdir):
    for file in fileList:
        if file != ".DS_Store":
            filedir = rootdir + "/" + file
            hdu = fits.open(filedir)
            # getting the flux and errors from the fits file
            # y = hdu[1].data.CORR_RATE
            # y_err = hdu[1].data.CORR_RATE_ERR
            y = hdu[1].data.SENSCORR_RATE
            y_err = hdu[1].data.SENSCORR_RATE_ERR
            x_err = hdu[1].data.TELAPSE / 2
            obsid = file[:11]
            ext = 1
            # determining the extension of the observation id
            if(file[12:-5] == "2_uvm2"):
                ext = 2
            if(file[12:-5] == "3_uvm2"):
                ext = 3
            # getting the times of the respective observation ids and graphing them

            fname = "/Users/shashankchavali/Downloads/AUMic_Swift_data/" + \
                    obsid+"/uvot/image/sw"+obsid+"um2_rw.img.gz"
            hdu2 = fits.open(fname)
            hdu2[ext].header
            start_time = Time(hdu2[ext].header["DATE-OBS"],
                              format="isot", scale="utc")
            end_time = Time(hdu2[ext].header["DATE-END"],
                            format="isot", scale="utc")
            t_mid_jd = np.mean([start_time.jd, end_time.jd])
            mean_time = Time(t_mid_jd, format="jd")
            x = mean_time.jd - 2458944.954386574
            x = x * 24 * 3600
            plt.errorbar(x, y, yerr=y_err, xerr=x_err, fmt="o", c="blue")

            ##### ALLISON'S ADDITION####
            time_array = np.append(time_array, mean_time.jd)
            isot_array = np.append(isot_array, mean_time.isot)
            exposure_time_array = np.append(
                exposure_time_array, hdu[1].data.TELAPSE)
            photometric_flux_array = np.append(photometric_flux_array, y)
            photometric_flux_error_array = np.append(
                photometric_flux_error_array, y_err)
            filename_array = np.append(filename_array, file)
            ra_array = np.append(ra_array, hdu2[ext].header['RA_PNT'])
            dec_array = np.append(dec_array, hdu2[ext].header['DEC_PNT'])
            roll_angle_array = np.append(
                roll_angle_array, hdu2[ext].header['PA_PNT'])

            ############################


##### ALLISON'S ADDITION ##############
# np.savetxt('/Users/aayoungb/Data/Swift/AUMic_grism_flares/AUMic_UVM2_photometry.txt',
#            np.transpose(np.array([time_array,exposure_time_array,photometric_flux_array,photometric_flux_error_array,
#                filename_array])),
#            header='time(jd),exptime(s),flux(counts/s),error(counts/s),filename',delimiter=',',fmt='%s')

df1 = pd.DataFrame(data={'Time (MJD)': time_array, 'Time (ISOT)': isot_array, 'RA (deg)': ra_array, 'Dec (deg)': dec_array,
                         'Roll Angle (deg)': roll_angle_array, 'exptime (s)': exposure_time_array,
                         'Flux (counts/s)': photometric_flux_array, 'Error (counts/s)': photometric_flux_error_array,
                         'filename': filename_array})
df1['Type'] = 'UVM2'
df1.to_csv(
    '/Users/shashankchavali/projects/Python/ResearchProject/AUMic_UVM2_photometry.txt')

########################################


plt.title("Simulated Photometric Flux")
plt.xlabel("Time from First Observation ID (s)")
plt.ylabel("Flux (counts/s)")
simulated_flux = Line2D([0], [0], marker='o', color='r')
uvm2_flux = Line2D([0], [0], marker='o', color='b')
handles = [simulated_flux, uvm2_flux]
labels = ['simulated flux', 'UVM2 flux']
plt.legend(handles=handles, labels=labels)
# plt.show()


###

df_both = pd.concat([df, df1])
df_both = df_both.sort_values('Time (MJD)')
df_both.to_csv(
    '/Users/shashankchavali/projects/Python/ResearchProject/AUMic_UVgrism_photometry_timesorted.csv')
print(np.mean(photometric_flux_array))
print(np.mean(photometric_flux_array))
print(np.mean(photometric_flux_array))
print(np.mean(photometric_flux_array))
print(np.mean(photometric_flux_array))