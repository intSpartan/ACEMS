import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np
import easyocr


# Path to the Tesseract executable (Update this with your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'two_faces.jpg'
image = cv2.imread(image_path)

# Step 1: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Bilateral Filtering to reduce noise while keeping edges sharp
bilateral_filtered = cv2.bilateralFilter(gray, 9, 75, 75)

# Step 3: Apply adaptive thresholding for better handling of uneven lighting
adaptive_thresh = cv2.adaptiveThreshold(bilateral_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Step 4: Apply contrast stretching to make text clearer without excessive contrast
# Get the min and max pixel values for contrast stretching
min_val = np.min(adaptive_thresh)
max_val = np.max(adaptive_thresh)

# Stretch contrast to full range [0, 255]
contrast_stretched = cv2.normalize(adaptive_thresh, None, 0, 255, cv2.NORM_MINMAX)

# Display the original and processed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(contrast_stretched, cmap='gray')
plt.title('Processed Image with Balanced Contrast')

plt.show()

# Save the processed image for OCR use
processed_image_path = 'processed_image_balanced.jpg'
cv2.imwrite(processed_image_path, contrast_stretched)

# Step 5: Extract text using pytesseract
extracted_text = pytesseract.image_to_string(contrast_stretched)
print("Extracted Text:")
print(extracted_text)

reader = easyocr.Reader(['en'])

# Perform OCR on the preprocessed image
results = reader.readtext(processed_image_path)

# Extract and print the text
extracted_text = ' '.join([result[1] for result in results])
print("Extracted Text from Preprocessed Image:")
print(extracted_text)