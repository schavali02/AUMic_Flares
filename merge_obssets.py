from PIL import Image
#Read the two images
image1 = Image.open('/Users/shashankchavali/Desktop/obs_set_6.png')
image2 = Image.open('/Users/shashankchavali/Desktop/obs_set_18.png')
new_image = Image.new('RGB',(image1.width, 2*image1.height), (250,250,250))
new_image.paste(image1,(0,0))
new_image.paste(image2, (0, image1.height))
new_image.save("/Users/shashankchavali/Desktop/paper_figure.jpg","JPEG")
new_image.show()