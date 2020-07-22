import numpy as np
from sklearn import cluster
from skimage import io
import matplotlib.pyplot as plt
from time import time

# Data/Image retrieval and pre-processing
def process_data():
    # URL links of three images
    URL1 = "https://raw.githubusercontent.com/jamesH-48/K-Means-A3/master/image2.jpg"
    URL2 = "https://raw.githubusercontent.com/jamesH-48/K-Means-A3/master/image3.jpg"
    URL3 = "https://raw.githubusercontent.com/jamesH-48/K-Means-A3/master/image5.jpg"

    # Read in images
    img1 = io.imread(URL1)
    img2 = io.imread(URL2)
    img3 = io.imread(URL3)

    # Store images in list
    images = [img1,img2,img3]
    return images

def color_quantization(image, kval, state):
    # Convert values for proper plt.imshow format
    img = np.array(image, dtype=np.float64) / 255
    w, h, d = img.shape
    re_img = np.reshape(img, (w*h, d))

    kmeans = cluster.KMeans(n_clusters=kval, random_state = state)
    labels = kmeans.fit_predict(re_img)
    palette = kmeans.cluster_centers_

    quan_image = np.reshape(palette[labels], (w,h,palette.shape[1]))
    return quan_image

def print_images(og_imgs, quan_imgs, num_imgs, kval):
    # Print Image Information
    col = 2
    row = num_imgs
    title = "Original Image vs Quantized Image \n k-value = " + str(kval)

    # Create Figure and set title
    fig, axs = plt.subplots(row,col)
    fig.suptitle(title, fontsize=24)

    # Plot Original Images
    axs[0,0].imshow(og_imgs[0])
    axs[1,0].imshow(og_imgs[1])
    axs[2,0].imshow(og_imgs[2])

    # Plot Quantized Images
    axs[0,1].imshow(quan_imgs[0])
    axs[1,1].imshow(quan_imgs[1])
    axs[2,1].imshow(quan_imgs[2])

    plt.show()

# Main
if __name__ == "__main__":
    # Set k-value
    kval = 16
    # Set state value
    state = 0
    # Set number of images
    img_num = 3
    # Initialize List of Quantized Images
    quantized_images = []
    # Get the three Images
    original_images = process_data()

    '''
    Perform Color Quantization
    '''
    # Set Time
    t0 = time()
    quan_image_1 = color_quantization(original_images[0], kval, state)
    print("Finished Image 1 Quantization in %0.3fs." % (time() - t0))
    # Set Time
    t0 = time()
    quan_image_2 = color_quantization(original_images[1], kval, state)
    print("Finished Image 2 Quantization in %0.3fs." % (time() - t0))
    # Set Time
    t0 = time()
    quan_image_3 = color_quantization(original_images[2], kval, state)
    print("Finished Image 3 Quantization in %0.3fs." % (time() - t0))
    
    # Gather Quantized Images
    quantized_images = [quan_image_1, quan_image_2, quan_image_3]
    # Print Image Comparisons
    print_images(original_images, quantized_images, img_num, kval)
