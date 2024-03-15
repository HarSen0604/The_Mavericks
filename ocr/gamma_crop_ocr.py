import cv2
import numpy as np
from PIL import Image
import easyocr
from datetime import datetime, timedelta
from word2number import w2n

# Global flag initialization
global_validation_flag = True

def remove_watermark(original_image, watermark_mask_path):
    gamma = 1.5
    gamma_corrected = np.uint8(cv2.pow(original_image / 255., gamma) * 255.)
    gamma_gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    _, binary_gamma_corrected = cv2.threshold(gamma_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_gamma_corrected_inverted = cv2.bitwise_not(binary_gamma_corrected)
    binary_gamma_image = binary_gamma_corrected_inverted
    watermark_image = cv2.imread(watermark_mask_path, cv2.IMREAD_GRAYSCALE)
    watermark_image = cv2.resize(watermark_image, (binary_gamma_image.shape[1], binary_gamma_image.shape[0]))
    watermark_image = 255 - watermark_image
    scale_factor = 4.0
    scaled_watermark = cv2.multiply(watermark_image, np.array([scale_factor]))
    filtered_image = cv2.subtract(binary_gamma_image, scaled_watermark)
    return filtered_image

def gamma_create(input_image_path, original_image):
    watermark_mask_path = 'c1.jpg'
    filtered_image = remove_watermark(original_image, watermark_mask_path)
    return filtered_image

def add_black_regions(image):
    pil_image = Image.fromarray(image)
    sizes =[(916,42,1235,90),(125,119,971,162),(200,156,1233,212),(89,208,829,265),(961,206,1232,264),(153,272,413,322),(874,375,1253,451),(338,496,971,551)]
    for i in sizes:
        x1, y1, x2, y2  = i
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                pil_image.putpixel((x, y), (0,))
    return np.array(pil_image)

def imgext(image1, image2):
    file_name = ['date', 'payee', 'amt1_txt', 'amt2_txt', 'amt_num', 'sign', 'accNo', 'bottom']
    sizes = [(916,42,1235,90),(125,119,971,162),(200,156,1233,212),(89,208,829,265),(961,206,1232,264),(874,375,1253,451)]
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
        
        # Processing for 'accNo' key
        elif key == 'accNo':
            result = en.readtext("accNo.jpg", detail=0, paragraph=True)
            ocr_dict['accNo'] = result[0]
        
        elif key == 'date':
            result = en.readtext("date.jpg", detail=0, paragraph=True)
            digits = ''.join(filter(str.isdigit, result[0]))
            digits = digits[0:2] + '-' + digits[2:4] + '-' + digits[4:]
            
            # date_obj = datetime.strptime(digits, '%d-%m-%Y')
            print(digits)
            date_obj = datetime.strptime(digits, '%d-%m-%Y')
            ocr_dict['date'] = date_obj

        else:
            result = en.readtext(f"{key}.jpg", detail=0, paragraph=True)
            ocr_dict[key] = result[0]
    
    return ocr_dict

def ocr_db(ocr_dict):
    global global_validation_flag
    local_validation_flag = True  # Initialize the local flag for this function
    ocr_dict1 = {}
    ocr_dict1['accNo'] = ocr_dict.get('accNo', '')
    ocr_dict1['payee'] = ocr_dict.get('payee', '')
    ocr_dict1['status'] = True  # Assuming validation passes initially
    ocr_dict1['reason'] = ''
    try:
        amt_num = w2n.word_to_num(ocr_dict.get('amt1_txt', '') + " " + ocr_dict.get('amt2_txt', ''))
        if amt_num != ocr_dict.get('amt_num', 0):
            ocr_dict1['reason'] = 'Amount in text and number do not match'
            local_validation_flag = False
            global_validation_flag = False
    except Exception as e:
        local_validation_flag = False
        global_validation_flag = False
        ocr_dict1['reason'] = str(e)
    ocr_dict1['amt_num'] = ocr_dict.get('amt_num', 0)
    ocr_dict1['chequeNo'] = ocr_dict.get('chequeNo', '')
    ocr_dict1['sign'] = Image.open('sign.jpg') if 'sign' in global_images else None
    return ocr_dict1, local_validation_flag

def validation(ocr_dict):
    global global_validation_flag
    local_validation_flag = True
    if 'date' in ocr_dict and ocr_dict['date'] is not None:
        given_date = ocr_dict['date']
        current_date = datetime.now()
        difference = current_date - given_date
        if difference > timedelta(days=90):
            local_validation_flag = False
            global_validation_flag = False
            ocr_dict['status'] = False
            ocr_dict['reason'] = "Validity of the check is expired"
    else:
        local_validation_flag = False
        global_validation_flag = False
    
    # use DB to check if cheque number is unique
    # use DB to check if accNo. and sign are same (integrate Snehan's code here)
    return local_validation_flag

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
    imgext(filtered_image, original_image)
    ocr_dict = ocr_model(hi_en, te_en, en)
    ocr_dict1, _ = ocr_db(ocr_dict)
    validation(ocr_dict1)
    if global_validation_flag:
        print("Validation Passed")
    else:
        print("Validation Failed")
    
    # then, push the final to db2
    # if status == true, take care of transactions

if __name__ == "__main__":
    main()
