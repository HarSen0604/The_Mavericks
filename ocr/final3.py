import cv2
import numpy as np
from PIL import Image
import pytesseract



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
input_image_path = 'tel.jpg'
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
cv2.imshow('Filtered blk Image', filtered_image_bl_reg)
cv2.waitKey(0)
cv2.destroyAllWindows()
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
print("No text in unwanted regions :",is_image_fully_black(filtered_image_bl_reg))




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


def calculate_pantograph_roughness(image_path):
    # Read the image in grayscale
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the gradients along the x-axis
    gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)

    # Take the absolute values of gradients
    abs_gradient_x = np.abs(gradient_x)

    # Sum the absolute gradients along the horizontal axis
    roughness_sum = np.sum(abs_gradient_x, axis=1)

    # Calculate the image size
    image_size = grayscale_image.shape[0] * grayscale_image.shape[1]

    # Normalize by the image size
    roughness = np.sum(roughness_sum) / image_size
    threshold = 45
    print("Roughness of the pantograph image(threshold based on cheque):", roughness)

    if threshold > roughness:
        print("It is genune.")
        return True
    else:
        print("It is not genuine.")
        return False
    
    

# Example usage
# image_path = 'first.jpg'



print("Acceptable roughness :",calculate_pantograph_roughness(input_image_path))




def fdmicrolines(img):
    def count_consecutive_pairs(text):
        consecutive_pairs = 0
        prev_char = None
        for char in text:
            if prev_char is not None and char == prev_char:
                consecutive_pairs += 1
            prev_char = char
        return consecutive_pairs

    
    gamma = 2
    # Perform OCR on the genuine microline image


    # Load the image as a NumPy array
    genuine_image = np.array(img)

    gamma_corrected = np.uint8(cv2.pow(genuine_image / 255., gamma) * 255.)

    # Ensure that the gamma-corrected image has 3 channels (for RGB images)
    if gamma_corrected.shape[-1] != 3:
        gamma_corrected = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2RGB)

    # Convert the gamma-corrected image to grayscale
    gamma_gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)

    # Convert the gamma-corrected image to a binary image using Otsu thresholding
    _, binary_gamma_corrected = cv2.threshold(gamma_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    genuine_text = pytesseract.image_to_string(cv2.bitwise_not(binary_gamma_corrected))


    # Count consecutive pairs of characters

    genuine_pairs_count = count_consecutive_pairs(genuine_text)

    threshold = 14
    print("Microline consecutive pairs count:", genuine_pairs_count)

    # Compare the counts
    if threshold < genuine_pairs_count:
        print("It is genune.")
        return True
    else:
        print("It is not genuine.")
        return False

    # Print the counts for reference
print("Acceptable microline consecutive pairs : ",fdmicrolines(original_image))