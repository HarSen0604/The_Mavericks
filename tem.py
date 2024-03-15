from flask import Flask, jsonify, request  # Add request import
from bson import ObjectId
from pymongo import MongoClient
from flask_cors import CORS
import json
import cv2
from flask import jsonify
import numpy as np
from PIL import Image
import pytesseract
import urllib.request
import easyocr
from datetime import datetime, timedelta
from word2number import w2n
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import requests
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)  


global globalurl 
MONGO_URI = "mongodb+srv://root:12345@cluster0.ljlbrlf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['Standardch']
stch_collection = db['stch']  

def check_mongo_connection():
    try:
        client.server_info()  
        print("MongoDB connected successfully!")
    except Exception as e:
        print("Failed to connect to MongoDB:", e)

check_mongo_connection()

@app.route('/', methods=['GET'])
def index():
    try:
        all_documents = stch_collection.find({})
        documents_json = []
        for doc in all_documents:
            doc['_id'] = str(doc['_id'])
            documents_json.append(doc)
        return jsonify(documents_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploadUrl', methods=['POST'])
def handle_post_request():
    data = request.json
    url = data.get('url')
    print(url)
    # If you don't need to perform any processing on the uploaded URL,
    # you can simply return a response indicating success
    return jsonify({'message': 'URL received successfully'})


def signature_tflag(acc_no):
    try:
        document = stch_collection.find_one({'Acc_No': acc_no})  # Find document by Acc_No
        if document:
            original_sign_url = document.get('original_sign')  # Get original_sign URL from document
            generated_sign_url = global_images[5]  # Replace 'URL_of_generated_signature' with actual URL
      
        else:
            print("Error: Document not found for Acc_No")
            return False, None

        def mse(imageA, imageB):
            if imageA is None or imageB is None:
                print("Error: One of the images is None.")
                return None

            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])
            return err

        def compare_img(imageA, imageB):
            m = mse(imageA, imageB)
            if m is None:
                return None
            s = ssim(imageA, imageB)
            a = s * 100
            return a

        def get_img_url(url):
            response = requests.get(url)
            if response.status_code == 200:
                img_data = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                return img
            else:
                print("Error: Failed to fetch image from URL")
                return None

        # Read the images
        generated_signature = get_img_url(generated_sign_url)
        original_signature = get_img_url(original_sign_url)

        # Check if images are loaded successfully
        if generated_signature is None or original_signature is None:
            print("Error: Unable to load image files.")
            return False, None

        # Convert the images to grayscale
        original_signature_gray = cv2.cvtColor(original_signature, cv2.COLOR_BGR2GRAY)
        generated_signature_gray = cv2.cvtColor(generated_signature, cv2.COLOR_BGR2GRAY)

        # Calculate the MSE and SSIM
        mse_score = mse(original_signature_gray, generated_signature_gray)
        ssim_score = compare_img(original_signature_gray, generated_signature_gray)

        if mse_score is None or ssim_score is None:
            print("Error: Unable to calculate MSE or SSIM.")
            return False, None

        # Calculate the confidence score
        signature_confidence_score = compare_img(original_signature_gray, generated_signature_gray)

        # Calculate the average score
        average_confidence_score = (ssim_score + signature_confidence_score) / 2

        print("Mean Squared Error:", mse_score)
        print("Structural Similarity Index:", ssim_score)
        print("Confidence Score:", signature_confidence_score)
        print("Average Score:", average_confidence_score)

        print("The Final Confidence Score between two images is %f percent" % average_confidence_score)

        if average_confidence_score  < 70.0:
            tflag = False
        else:
            tflag = True

        return tflag, average_confidence_score
    except Exception as e:
        return False, str(e)




errs={}

def remove_watermark(original_image, watermark_image):
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
    watermark_image = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)

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

def get_img_url(url):
    # url = "https://res.cloudinary.com/dbq6389ro/image/upload/v1710482421/z3guh7n20r3tpah8mq8e.jpg"

    # Download the image
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img




def add_black_regions(image):
    pil_image = Image.fromarray(image)
    sizes =[(916,42,1235,90),(122,100,976,172),(200,170,1233,222),(89,208,829,275),(961,206,1232,264),(153,272,413,322),(874,375,1253,451),(338,496,971,551)]

    for i in sizes:
        x1, y1, x2, y2  = i  # Top-left corner

        # Iterate through the region defined by the coordinates and set pixels to black
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                pil_image.putpixel((x, y), (0,))  # Set pixel to black (RGB: 0, 0, 0)

    return np.array(pil_image)

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



def set_flag_true():
    global tflag
    tflag = True  # Set the flag to True



def imgext(image1, image2):
    file_name = ['date', 'payee', 'amt1_txt', 'amt2_txt', 'amt_num', 'sign', 'accNo', 'bottom']
    sizes=[(916,42,1235,90),(122,100,976,172),(200,170,1233,222),(89,208,829,275),(961,206,1232,264),(874,375,1253,451)]
    sizes1 = [(153,272,413,322),(338,496,971,551)]
    a = 0
    global global_images
    global_images = {}
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    for i in sizes:
        temp = (i[2] - i[0], i[3] - i[1])
        new = Image.new('RGB',temp,color='white')
        imm = Image.fromarray(image1_rgb)
        cropped = imm.crop(i)
        new.paste(cropped)
        new_filename = f"{file_name[a]}.jpg"
        new.save(new_filename)
        global_images[f"{file_name[a]}"] = new
        a += 1
    for i in sizes1:
        temp = (i[2] - i[0], i[3] - i[1])
        new = Image.new('RGB',temp,color='white')
        imm = Image.fromarray(image2_rgb)
        cropped = imm.crop(i)
        new.paste(cropped)
        new_filename = f"{file_name[a]}.jpg"
        new.save(new_filename)
        global_images[f"{file_name[a]}"] = new
        a += 1




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


def modify_status(acc_no, new_status):
    try:
        # Update the document in the collection
        result = stch_collection.update_one(
            {'Acc_No': acc_no},
            {'$set': {'status': new_status}}
        )

        
    except Exception as e:
        print(e)






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


def ocr_model(hi_en, te_en, en):
    ocr_dict = {}
    for key, value in global_images.items():
        print(key)
        if key == 'sign':
            continue

        elif key in ['payee', 'amt1_txt', 'amt2_txt']:
            hi_en_1 = hi_en.readtext(f"{key}.jpg")
            te_en_1 = te_en.readtext(f"{key}.jpg")
            print(hi_en_1)
            print(te_en_1)
            if key == 'amt2_txt':
                if len(hi_en_1) == 0 or len(te_en_1) == 0:
                    ocr_dict[key] = ''
                    print(ocr_dict[key])
                    continue

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
        
        elif key == 'date':
            result = en.readtext("date.jpg", detail=0, paragraph=True)
            digits = ''.join(filter(str.isdigit, result[0]))
            digits = digits[0:2] + '-' + digits[2:4] + '-' + digits[4:]
            
            # date_obj = datetime.strptime(digits, '%d-%m-%Y')
            print(digits)
            try:
                date_obj = datetime.strptime(digits, "%d-%m-%Y")
                errs['invalid-date'] = "False"
                # Proceed with further processing using date_obj
            except ValueError as e:
                current_date = datetime.now()
                date_obj = current_date.strftime("%d-%m-%Y")
                errs['invalid-date'] = "True"

                print("Error parsing date:", e)
            ocr_dict['date'] = date_obj

        else:
            result = en.readtext(f"{key}.jpg", detail=0, paragraph=True)
            ocr_dict[key] = result[0]
    
    return ocr_dict

def ocr_db(ocr_dict):
    
    local_validation_flag = True  # Initialize the local flag for this function
    ocr_dict1 = {}
    ocr_dict1['accNo'] = ocr_dict.get('accNo', '')
    ocr_dict1['payee'] = ocr_dict.get('payee', '')
    ocr_dict1['amt_num'] = float(ocr_dict['amt_num'].replace('/', '').replace('-', '').replace('/-', '').replace(' ', ''))
    ocr_dict1['chequeNo'] = ocr_dict.get('chequeNo', '')
    ocr_dict1['sign'] = Image.open('sign.jpg') if 'sign' in global_images else None
    ocr_dict1['status'] = True  # Assuming validation passes initially
    #ocr_dict1['cheque_img'] Upload the image with this dict da
    ocr_dict1['reason'] = ''
    ocr_dict1['date'] = ocr_dict['date']
    return ocr_dict1, local_validation_flag

def validation(ocr_dict):
    global global_validation_flag
    local_validation_flag = True
    if 'date' in ocr_dict and ocr_dict['date'] is not None:
        given_date = ocr_dict['date']
        print(given_date)
        current_date = datetime.now()
        difference = current_date - given_date
        if difference > timedelta(days=90):
            local_validation_flag = False
            global_validation_flag = False
            ocr_dict['status'] = False
            ocr_dict['reason'] = "Validity of the check is expired"
            print("Validity of the check is expired")
    else:
        given_date = ocr_dict['date']
        print("given date: " +given_date)
        print("Date is failed")
        local_validation_flag = False
        global_validation_flag = False
    
    # use DB to check if cheque number is unique
    # use DB to check if accNo. and sign are same (integrate Snehan's code here)
    return local_validation_flag, ocr_dict


def update_reason_by_acc_no(acc_no, new_reason):
    try:
        # Update the document in the collection
        result = stch_collection.update_one(
            {'Acc_No': acc_no},
            {'$set': {'reason': new_reason}}
        )

        if result.modified_count == 1:
            print('Reason updated successfully : ')
            return True
        else:
            print('Failed to update reason, Acc_No not found')
            return False
    except Exception as e:
        print("Error:", e)
        return False




def mn(inpt_url):
    global hi_en
    global te_en
    global en
    global global_validation_flag 

    hi_en = easyocr.Reader(['hi','en'], gpu=True)
    te_en = easyocr.Reader(['te','en'], gpu=True)
    en = easyocr.Reader(['en'], gpu=True)


    # Example usage:
    input_image_path = 'cw.jpg'
    original_image = cv2.imread(input_image_path)


    original_image = get_img_url(inpt_url)


    # watermark_mask_path = 'c1.jpg'
    watermark_mask_img = get_img_url("https://res.cloudinary.com/dbq6389ro/image/upload/v1710485366/wys066svvn2eazvrn3iz.jpg")
    filtered_image = remove_watermark(original_image, watermark_mask_img)

    # Show or process the filtered image as needed
    # cv2.imshow('Filtered Image', filtered_image)

    # cv2.imshow('mask Image', watermark_mask_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    filtered_image_bl_reg = add_black_regions(filtered_image)
    # cv2.imshow('Filtered blk Image', filtered_image_bl_reg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    tem = is_image_fully_black(filtered_image_bl_reg)
    print("No text in unwanted regions :",tem)
    if tem:
        errs["unwanted-text"] = "False"
        
    else : 
        global_validation_flag = False
        errs["unwanted-text"] = "True"
    imgext(filtered_image, original_image)

    tem = calculate_pantograph_roughness(input_image_path)
    print("Acceptable roughness :",tem)

    if tem:
        errs['Accepted_roughness'] = "True"
    else:
        global_validation_flag = False
        errs['Accepted_roughness'] = "False"

    tem = fdmicrolines(original_image)
    print("Acceptable microline consecutive pairs : ",tem)
    if tem:
        errs['Accepted_microline'] = "True"
    else:
        errs['Accepted_microline'] = "False"
        global_validation_flag = False


    ocr_dict1 = None
    try:
        ocr_dict = ocr_model(hi_en, te_en, en)
        ocr_dict1, _ = ocr_db(ocr_dict)
        _, ocr_dict1 = validation(ocr_dict1)
    except Exception as e:
        print(e)
        global_validation_flag = False
    


    acc_no = "FDRL121234568"  # Set Acc_No value
    
    # tflag, average_confidence_score = signature_tflag(ocr_dict1["accNo"])
    tflag, average_confidence_score = signature_tflag(acc_no)

    # Output the result
    if average_confidence_score is not None:
        print("tflag:", tflag)
        print("Average Confidence Score:", average_confidence_score)
    if not tflag:
        global_validation_flag = False

    modify_status(acc_no, global_validation_flag)
    print(global_validation_flag)


    

    errs["Sign_match"] = str(tflag)
    print("err",errs)
    dict_str = json.dumps(errs)
    if ocr_dict1:
        ocr_dict1['reason'] = dict_str
    update_reason_by_acc_no(acc_no,dict_str)
    print("\n\nOCR: ")
    print(ocr_dict1)
    return global_validation_flag




@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Receive the data from the frontend
        data = request.json

        # Call the function to process the image
        result = mn(data)

        # Check the result and return success or failure accordingly
        if result:
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "error": "Failed to process the image"}), 500
    except Exception as e:
            return jsonify({"Error":"erroe"})













app.run(debug=True)