#First part of the reduction

#Run this code in a loop over the entire data directory and it will reduce all of the data
ra, dec = 311.28971820645245, -31.34090047520722
obsid = "00095500062"
datadir = "/Users/shashankchavali/Downloads/AUMic_Swift_data/00095500062/uvot/image"
ext = 1
uvotgetspec.getSpec(ra,dec,obsid,ext,indir=datadir,fit_second=True,chatter=1,skip_field_src=True,wheelpos=160)


#Second part of the reduction

#This part takes the output products of the previous part of the reduction and turns it into a more manageable form
#The output file is a file which consists of 5 columns: wavelength, weighted flux, variance weighted flux, flux, flux error deviations from mean,
#flux error mean noise, number of data summed, and sector
#This form is easy to read into a pandas dataframe and work with


pha = ['/Users/shashankchavali/projects/Python/ResearchProject/sw00095500048ugu_1ord_2_f.pha']
out = '00095500048_2.txt'
uvotspec.sum_PHAspectra(pha, outfile = out)

