import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def signature_tflag():

    def mse(imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    def compare_img(imageA, imageB):
        m = mse(imageA, imageB)
        s = ssim(imageA, imageB)
        a = s * 100
        return a


    # Convert the images to grayscale

    generated_signature = global_images[6]  # Assuming the signature is stored at index 0
    original_signature = cv2.imread("path of sign derived from mongo db", cv2.IMREAD_GRAYSCALE)

    # Calculate the MSE and SSIM
    mse_score = mse(original_signature, generated_signature)
    ssim_score = ssim(original_signature, generated_signature)

    # Calculate the confidence score
    signature_confidence_score = compare_img(original_signature, generated_signature)

    # Calculate the average score
    average_confidence_score = (mse_score + ssim_score + signature_confidence_score) / 3

    print("Mean Squared Error:", mse_score)
    print("Structural Similarity Index:", ssim_score)
    print("Confidence Score:", signature_confidence_score)
    print("Average Score:", average_confidence_score)

    print("The Final Confidence Score between two images is %f percent" % average_confidence_score)

    if average_confidence_score < 75.0:
        tflag = False
    else:
        tflag = True
    
    return tflag


