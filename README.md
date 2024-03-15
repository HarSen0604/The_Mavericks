# The_Mavericks
Standard Chartered GBS Campus Engagement Event 2024
# Automated Cheque Clearing System

## Introduction

This project aims to automate the cheque clearing process for banks, particularly focusing on reducing human intervention, processing time, and potential fraud detection. The system utilizes various AI/ML techniques along with Optical Character Recognition (OCR) and Image Processing to streamline the clearing process.

## Problem Statement

In the current scenario, banks handle large volumes of cheques in the clearing process, involving numerous manual steps such as signature verification, which consume significant human resources and time. The manual verification process is prone to errors and delays.

## Approach

### 1. Automation using AI/ML/ICR/OCR

- **Automatic Data Entry & Technical Verification**: The system automates data entry and technical verification processes by leveraging AI/ML algorithms. Batch processing is done based on the cheque's date, reducing manual effort.

- **Signature Verification**: Signature verification is performed using Mean Squared Error (MSE) and Structural Similarity Index (SSIM) techniques. A confidence score is calculated based on SSIM, indicating the similarity between the original and generated signatures. If the confidence score is below a threshold (75.0), potential discrepancies are flagged for manual inspection.

### 2. Multilingual Support

The system supports multiple languages including English, Hindi, and Telugu, enhancing its usability across diverse regions.

### 3. Effort and Time Reduction

- **User-friendly Interface**: The system features a simple interface with buttons for initiating processes, reducing the need for extensive training.

- **Optimized Processing**: Once the OCR is loaded, it remains in memory for subsequent cheque images, minimizing processing time.

### 4. Fraud Detection

The system employs advanced techniques to detect potential frauds such as overwriting, scribbling, and the use of void pantographs. Void pantographs create copy-evident backgrounds on cheques, enabling the system to identify fake or altered documents.

## Technologies Used
Backend:
- Flask: Python-based web framework used for creating the backend server.
- OpenCV: Library for computer vision tasks such as image processing and analysis.
- NumPy: Python library for numerical computations, often used for handling arrays and matrices.
- Pillow: Python Imaging Library (PIL) fork, used for opening, manipulating, and saving many different image file formats.
- Pytesseract: Python wrapper for Google's Tesseract-OCR Engine, used for optical character recognition (OCR).
- EasyOCR: Python package for optical character recognition (OCR) with support for multiple languages.
- Word2Number: Python package for converting text numbers (words) into numerical values.
- scikit-image: Collection of algorithms for image processing tasks such as image enhancement, segmentation, and feature extraction.
- Requests: Python library for making HTTP requests.

Database:
- PyMongo: Python driver for MongoDB, used for interacting with MongoDB databases.

Frontend (if React and JavaScript are involved):
- React: JavaScript library for building user interfaces.
- JavaScript: Programming language commonly used for web development.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HarSen0604/The_Mavericks.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure that the scanned cheque images are of resolution 300 dpi.

2. Run the application:
   ```bash
   python main.py
   ```

3. Use the provided interface to initiate the clearing process and perform signature verification.

## Future Improvements

- Integration with banking systems for seamless cheque processing.
- Usage of high - end scanners for more precise fraud detection
- Enhancement of fraud detection algorithms to cover more sophisticated techniques.
- Implementation of real-time processing for immediate cheque clearance.

## Contributors
- Students from PSG College of Technology, BE CSE III Year, (2021-25)
- Akshay Perison Davis (21Z205) @Akshay1805
- Hareesh S (21Z218) @HarSen0604
- Kavin Dev R (21Z224) @kavinDEV15
- Navaneetha Krishnan K S (21Z233) @navaneeenavan
- Sharan S (21Z254) @sharanzzgit
- Snehan E M (21Z257) @sanbec234
