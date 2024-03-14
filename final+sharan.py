import cv2
import numpy as np
from PIL import Image


def remove_watermark(original_image, watermark_mask_path):
    # Load the original RGB image

    # Apply gamma correction
    gamma = 1.5
    gamma_corrected = np.uint8(cv2.pow(original_image / 255., gamma) * 255.)

    # Convert the gamma-corrected image to grayscale
    gamma_gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)

    # Convert the gamma-corrected image to a binary image using Otsu thresholding
    _, binary_gamma_corrected = cv2.threshold(gamma_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the binary image
    binary_gamma_corrected_inverted = cv2.bitwise_not(binary_gamma_corrected)

    # Load the binary gamma image and the watermark image
    binary_gamma_image = binary_gamma_corrected_inverted
    watermark_image = cv2.imread(watermark_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same dimensions
    watermark_image = cv2.resize(watermark_image, (binary_gamma_image.shape[1], binary_gamma_image.shape[0]))

    # Invert watermark image (if necessary)
    watermark_image = 255 - watermark_image

    # Scale the values of the watermark mask to increase the amount of watermark removed
    scale_factor = 4.0  # Adjust this value as needed
    scaled_watermark = cv2.multiply(watermark_image, np.array([scale_factor]))

    # Apply watermark removal
    filtered_image = cv2.subtract(binary_gamma_image, scaled_watermark)

    return filtered_image

# Example usage:
input_image_path = 'c4.jpg'
original_image = cv2.imread(input_image_path)
watermark_mask_path = 'c1.jpg'
filtered_image = remove_watermark(original_image, watermark_mask_path)

# Show or process the filtered image as needed
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




def add_black_regions(image):
    pil_image = Image.fromarray(image)
    sizes =[(916,42,1235,90),(125,119,971,162),(200,156,1233,212),(89,208,829,265),(961,206,1232,264),(153,272,413,322),(874,375,1253,451),(338,496,971,551)]

    for i in sizes:
        x1, y1, x2, y2  = i  # Top-left corner

        # Iterate through the region defined by the coordinates and set pixels to black
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                pil_image.putpixel((x, y), (0,))  # Set pixel to black (RGB: 0, 0, 0)

    return np.array(pil_image)

filtered_image_bl_reg = add_black_regions(filtered_image)

def is_image_fully_black(image):
    # Load the image
    # image_path = "modified_image.jpg"
    # Check if the image is loaded successfully
    if image is None:
        print("Error: Could not load image.")
        return False

    # Check if all pixels are black (0)
    if (image == 0).all():
        return True
    else:
        return False
print(is_image_fully_black(filtered_image_bl_reg))




def set_flag_true():
    global tflag
    tflag = True  # Set the flag to True

def imgext(image1, image2):
    # Define the sizes for cropping images
    sizes = [(916,42,1235,90),(125,119,971,162),(200,156,1233,212),(89,208,829,265),(961,206,1232,264),(874,375,1253,451)]
    sizes1 = [(153,272,413,322),(338,496,971,551)]
    a = 1
    b = 1
    global global_images  # Declare global_images outside the loop
    global_images = []  # Initialize the list

    # Read the images using OpenCV
    # Convert the images to RGB format
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Process the first image
    for i in sizes:
        new = Image.new('RGB',(1300,150),color='white')  # Create a new white image
        imm = Image.fromarray(image1_rgb)  # Convert numpy array to PIL Image
        cropped = imm.crop(i)  # Crop the image based on size
        new.paste(cropped, (350,50))  # Paste the cropped image onto the new image
        new_filename = f"10{b}_{a}.jpg"  # Define the filename for the new image
        new.save(new_filename)  # Save the new image
        global_images.append(new)  # Append the new image to the global list
        a += 1
    b += 1

    # Process the second image
    for i in sizes1:
        new = Image.new('RGB',(1300,150),color='white')  # Create a new white image
        imm = Image.fromarray(image2_rgb)  # Convert numpy array to PIL Image
        cropped = imm.crop(i)  # Crop the image based on size
        new.paste(cropped, (350,50))  # Paste the cropped image onto the new image
        new_filename = f"10{b}_{a}.jpg"  # Define the filename for the new image
        new.save(new_filename)  # Save the new image
        global_images.append(new)  # Append the new image to the global list
        a += 1
    b += 1

    # Display the processed images
    for idx, image in enumerate(global_images, start=1):
        image.show()

imgext(filtered_image, original_image)