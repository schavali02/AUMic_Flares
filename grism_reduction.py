#First part of the reduction
ra, dec = 311.28971820645245, -31.34090047520722
obsid = "00095500062"
datadir = "/Users/shashankchavali/Downloads/AUMic_Swift_data/00095500062/uvot/image"
ext = 1
uvotgetspec.getSpec(ra,dec,obsid,ext,indir=datadir,fit_second=True,chatter=1,skip_field_src=True,wheelpos=160)


#Second part of the reduction
pha = ['/Users/shashankchavali/projects/Python/ResearchProject/sw00095500048ugu_1ord_2_f.pha']
out = '00095500048_2.txt'
uvotspec.sum_PHAspectra(pha, outfile = out)

