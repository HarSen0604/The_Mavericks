import cv2
import numpy as np
from PIL import Image
import easyocr
from datetime import datetime, timedelta
from word2number import w2n

def remove_watermark(original_image, watermark_mask_path):
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

def gamma_create(input_image_path, original_image):
    watermark_mask_path = 'c1.jpg'
    filtered_image = remove_watermark(original_image, watermark_mask_path)

    # Show or process the filtered image as needed
    # cv2.imshow('Filtered Image', filtered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return filtered_image

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

def is_image_fully_black(image):
    if image is None:
        print("Error: Could not load image.")
        return False

    # Check if all pixels are black (0)
    if (image == 0).all():
        return True
    else:
        return False
    
def set_flag_true():
    global tflag
    tflag = True  # Set the flag to True

def imgext(image1, image2):
    # Define the sizes for cropping images
    file_name = ['date', 'payee', 'amt1_txt', 'amt2_txt', 'amt_num', 'sign', 'accNo', 'bottom']
    sizes = [(916,42,1235,90),(125,119,971,162),(200,156,1233,212),(89,208,829,265),(961,206,1232,264),(874,375,1253,451)]
    sizes1 = [(153,272,413,322),(338,496,971,551)]
    a = 0
    b = 1

    global global_images  # Declare global_images outside the loop
    global_images = {}  # Initialize the dictionary

    # Read the images using OpenCV
    # Convert the images to RGB format
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Process the first image
    for i in sizes:
        temp = (i[2] - i[0], i[3] - i[1])
        new = Image.new('RGB',temp,color='white')  # Create a new white image
        imm = Image.fromarray(image1_rgb)  # Convert numpy array to PIL Image
        cropped = imm.crop(i)  # Crop the image based on size
        new.paste(cropped)  # Paste the cropped image onto the new image
        new_filename = f"{file_name[a]}.jpg"  # Define the filename for the new image
        new.save(new_filename)  # Save the new image
        global_images[f"{file_name[a]}"] = new  # Add the new image to the global dictionary with a unique key
        a += 1

    # Process the second image
    for i in sizes1:
        temp = (i[2] - i[0], i[3] - i[1])
        new = Image.new('RGB',temp,color='white')  # Create a new white image
        imm = Image.fromarray(image2_rgb)  # Convert numpy array to PIL Image
        cropped = imm.crop(i)  # Crop the image based on size
        new.paste(cropped)  # Paste the cropped image onto the new imagef
        new_filename = f"{file_name[a]}.jpg"  # Define the filename for the new image
        new.save(new_filename)  # Save the new image
        global_images[f"{file_name[a]}"] = new  # Add the new image to the global dictionary with a unique key
        a += 1

def ocr_model(hi_en, te_en, en):
    ocr_dict = {}
    
    for key, value in global_images.items():
        if key == 'sign':
            pass

        elif key in ['payee', 'amt1_txt', 'amt2_txt']:
            hi_en_1 = hi_en.readtext(f"{key}.jpg")
            te_en_1 = te_en.readtext(f"{key}.jpg")
            avg_confidence_data1 = sum(item[-1] for item in hi_en_1) / len(hi_en_1)
            avg_confidence_data2 = sum(item[-1] for item in te_en_1) / len(te_en_1)
            result = ''
            if avg_confidence_data1 > avg_confidence_data2:
                result = hi_en.readtext(f"{key}.jpg", detail=0, paragraph=True)
            elif avg_confidence_data1 < avg_confidence_data2:
                result = te_en.readtext(f"{key}.jpg", detail=0, paragraph=True)
            ocr_dict[key] = result[0]
        
        elif key == 'bottom':
            result = en.readtext("bottom.jpg", detail=0, paragraph=True)
            chequeNo = ''
            leng = 0
            for i in result[0]:
                if leng == 6:
                    break
                if i.isdigit():
                    chequeNo += i
                    leng += 1
            ocr_dict['chequeNo'] = chequeNo
        
        # Processing for 'accNo' key
        elif key == 'accNo':
            result = en.readtext("accNo.jpg", detail=0, paragraph=True)
            ocr_dict['accNo'] = result[0]
        
        elif key == 'date':
            result = en.readtext("date.jpg", detail=0, paragraph=True)
            digits = ''.join(filter(str.isdigit, result[0]))
            digits = digits[0:2] + '-' + digits[2:4] + '-' + digits[4:]
            
            # date_obj = datetime.strptime(digits, '%d-%m-%Y')
            date_obj = datetime.strptime('15-03-2024', '%d-%m-%Y')
            ocr_dict['date'] = date_obj

        else:
            result = en.readtext(f"{key}.jpg", detail=0, paragraph=True)
            ocr_dict[key] = result[0]
    
    return ocr_dict
        
def ocr_db(ocr_dict):
    bool = False
    ocr_dict1 = {}
    file_name = ['date', 'payee', 'amt1_txt', 'amt2_txt', 'amt_num', 'sign', 'accNo', 'bottom']
    
    ocr_dict1['accNo'] = ocr_dict['accNo']
    ocr_dict1['payee'] = ocr_dict['payee']
    ocr_dict1['status'] = False # Initially, false
    ocr_dict1['reason'] = ''

    str_ = ocr_dict['amt1_txt'] + " " + ocr_dict['amt2_txt']
    if (w2n.word_to_num(str) == ocr_dict['amt_num']): # amt1_txt + " " + amt2_txt
        ocr_dict1['amt_num'] = ocr_dict['amt_num']
    else:
        ocr_dict1['reason'] = 'Amount in text and number do not match'
        bool = False

    ocr_dict1['chequeNo'] = ocr_dict['chequeNo']
    ocr_dict1['sign'] = Image.open('sign.jpg')

    return ocr_dict1, bool

    # ocr_dict['amt_txt'] = 'Ninety thousand only' 
    # ocr_dict['cheque'] = Image.open('c4.jpg') already exists in the DB. Just for reference
    # ocr_dict['date'] = datetime.strptime('15-03-2024', "%d-%m-%Y") For reference  

def validation(ocr_dict):
    # Check if the difference is less than or equal to 90 days
    given_date = datetime.strptime(ocr_dict['date'], "%Y-%m-%d")
    current_date = datetime.now()
    difference = current_date - given_date
    if difference <= timedelta(days=90):
        ocr_dict['status'] = False
        ocr_dict['reason'] = "Validity of the check is expired"
        return
    
    # use DB to check if cheque number is unique
    # use DB to check if accNo. and sign are same (integrate Snehan's code here)

def main():
    global hi_en
    global te_en
    global en
    hi_en = easyocr.Reader(['hi','en'], gpu=True)
    te_en = easyocr.Reader(['te','en'], gpu=True)
    en = easyocr.Reader(['en'], gpu=True)
    input_image_path = 'c4.jpg'
    original_image = cv2.imread(input_image_path)
    filtered_image = gamma_create(input_image_path, original_image)
    filtered_image_bl_reg = add_black_regions(filtered_image)
    print(is_image_fully_black(filtered_image_bl_reg))
    imgext(filtered_image, original_image)

    ocr_dict = ocr_model(hi_en, te_en, en)
    print(ocr_dict)

    ocr_dict1, bool = ocr_db(ocr_dict)
    print(ocr_dict1)

    if (bool):
        validation(ocr_dict1)

    # then, push the final to db2
    # if status == true, take care of transactions


if __name__ == "__main__":
    main()
