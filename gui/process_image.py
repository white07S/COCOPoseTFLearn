import cv2
import numpy as np
import matplotlib.pyplot as plt
import util  
import config_reader 

def process_image(oriImg, model):
    param, model_params = config_reader.config_reader()
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    # first figure shows padded images
    f, axarr = plt.subplots(1, len(multiplier))
    f.set_size_inches((20, 5))
    # second figure shows heatmaps
    f2, axarr2 = plt.subplots(1, len(multiplier))
    f2.set_size_inches((20, 5))
    # third figure shows PAFs
    f3, axarr3 = plt.subplots(2, len(multiplier))
    f3.set_size_inches((20, 10))


    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.pad_right_down_corner(imageToTest, model_params['stride'], model_params['padValue'])        
        axarr[m].imshow(imageToTest_padded[:,:,[2,1,0]])
        axarr[m].set_title('Input image: scale %d' % m)

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
        print("Input shape: " + str(input_img.shape))  

        output_blobs = model.predict(input_img)
        print("Output shape (heatmap): " + str(output_blobs[1].shape))

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # visualization
        axarr2[m].imshow(oriImg[:,:,[2,1,0]])
        ax2 = axarr2[m].imshow(heatmap[:,:,3], alpha=.5) # right elbow
        axarr2[m].set_title('Heatmaps (Relb): scale %d' % m)

        axarr3.flat[m].imshow(oriImg[:,:,[2,1,0]])
        ax3x = axarr3.flat[m].imshow(paf[:,:,16], alpha=.5) # right elbow
        axarr3.flat[m].set_title('PAFs (x comp. of Rwri to Relb): scale %d' % m)
        axarr3.flat[len(multiplier) + m].imshow(oriImg[:,:,[2,1,0]])
        ax3y = axarr3.flat[len(multiplier) + m].imshow(paf[:,:,17], alpha=.5) # right wrist
        axarr3.flat[len(multiplier) + m].set_title('PAFs (y comp. of Relb to Rwri): scale %d' % m)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    f2.subplots_adjust(right=0.93)
    cbar_ax = f2.add_axes([0.95, 0.15, 0.01, 0.7])
    _ = f2.colorbar(ax2, cax=cbar_ax)

    f3.subplots_adjust(right=0.93)
    cbar_axx = f3.add_axes([0.95, 0.57, 0.01, 0.3])
    _ = f3.colorbar(ax3x, cax=cbar_axx)
    cbar_axy = f3.add_axes([0.95, 0.15, 0.01, 0.3])
    _ = f3.colorbar(ax3y, cax=cbar_axy)
    # Return the figures for displaying in Streamlit
    return f, f2, f3, paf_avg, heatmap_avg


