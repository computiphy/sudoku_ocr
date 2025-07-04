import cv2
import numpy as np
from tensorflow.keras.models import load_model


#### READ THE MODEL WEIGHTS
def intializePredectionModel():
    model = load_model('core/models/mnist_model.h5"')
    return model


#### 1 - Preprocessing Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold


#### 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#### 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


#### 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


#### 4 - GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes,model):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = 1 - (img / 255)
        img = img.reshape(1, 784)
        ## GET PREDICTION
        predictions = model.predict(img)
        # classIndex = model.predict_classes(img)
        classIndex = np.argmax(predictions,axis=1)
        probabilityValue = np.amax(predictions)
        ## SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


# def getPredection(boxes, model, confidence_threshold=0.9):
#     """
#     Predicts digits for a list of extracted cell images (boxes) using a trained model.
#     Includes advanced preprocessing to isolate and center digits within each cell,
#     and more robust empty cell detection.

#     Args:
#         boxes (list of np.array): A list of individual image arrays, where each
#                                    array represents a potential digit cell. These
#                                    are expected to be grayscale or BGR images.
#         model (tf.keras.Model): The pre-trained Keras model for digit recognition.
#         confidence_threshold (float): The minimum probability value required for a
#                                       prediction to be considered a valid digit.
#                                       If below this, the cell is classified as 0 (empty).

#     Returns:
#         list: A list of predicted digits (0-9), where 0 indicates an empty cell.
#     """
#     result = []
#     model_input_size = (28, 28) # Assuming your MNIST model expects 28x28 input

#     # Parameters for digit isolation and filtering
#     min_contour_area_ratio = 0.02 # Minimum contour area as a percentage of cell area
#     max_contour_area_ratio = 0.6 # Maximum contour area as a percentage of cell area (to exclude large noise/lines)
#     min_aspect_ratio = 0.1 # Minimum width/height ratio for a digit contour
#     max_aspect_ratio = 10.0 # Maximum width/height ratio for a digit contour
#     min_pixel_intensity_for_digit = 20 # Minimum average pixel intensity (0-255) in the cropped digit area

#     for image in boxes:
#         img_cell = np.asarray(image)

#         # 1. Convert to grayscale if it has color channels
#         if len(img_cell.shape) == 3 and img_cell.shape[2] == 3:
#             img_cell = cv2.cvtColor(img_cell, cv2.COLOR_BGR2GRAY)
#         elif len(img_cell.shape) == 3 and img_cell.shape[2] == 1:
#             # Already grayscale with a channel dimension, remove it for processing
#             img_cell = img_cell.squeeze()

#         # Apply adaptive thresholding to highlight the digit
#         # This is crucial for separating digit from background within the cell
#         # THRESH_BINARY_INV means white digits on a black background, common for MNIST models
#         img_thresh = cv2.adaptiveThreshold(img_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                            cv2.THRESH_BINARY_INV, 11, 2)

#         # --- Debugging Tip: Uncomment to visualize thresholded cell ---
#         # cv2.imshow("Thresholded Cell", img_thresh)
#         # cv2.waitKey(0)

#         # Find contours in the thresholded image
#         contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         digit_found_in_cell = False
#         if len(contours) > 0:
#             # Find the largest contour, which should ideally be the digit
#             largest_contour = max(contours, key=cv2.contourArea)
#             contour_area = cv2.contourArea(largest_contour)
#             x, y, w, h = cv2.boundingRect(largest_contour)
            
#             cell_area = img_cell.shape[0] * img_cell.shape[1]

#             # Filter out very small contours (noise) or very large ones (grid lines)
#             # Also filter by aspect ratio to exclude non-digit shapes
#             if (contour_area > (cell_area * min_contour_area_ratio) and
#                 contour_area < (cell_area * max_contour_area_ratio) and
#                 (float(w)/h > min_aspect_ratio and float(w)/h < max_aspect_ratio)):

#                 # Add a small padding to the bounding box to ensure digit is not cut off
#                 padding = 3
#                 x_padded = max(0, x - padding)
#                 y_padded = max(0, y - padding)
#                 w_padded = min(img_cell.shape[1] - x_padded, w + 2 * padding)
#                 h_padded = min(img_cell.shape[0] - y_padded, h + 2 * padding)

#                 # Crop the original grayscale cell image to isolate the digit
#                 cropped_digit = img_cell[y_padded : y_padded + h_padded, x_padded : x_padded + w_padded]

#                 # --- Debugging Tip: Uncomment to visualize cropped digit ---
#                 # cv2.imshow("Cropped Digit", cropped_digit)
#                 # cv2.waitKey(0)

#                 # Early exit for truly empty cells that might have small noise contours
#                 # Check if the cropped area is mostly black (assuming white digit on black background after thresholding)
#                 if np.mean(cropped_digit) < min_pixel_intensity_for_digit: # Adjust this value if needed
#                     result.append(0)
#                     continue # Skip to next box

#                 # Create a black canvas of the model's input size
#                 final_img = np.zeros(model_input_size, dtype=np.uint8)

#                 # Resize the cropped digit while maintaining aspect ratio
#                 # Then place it in the center of the black canvas
#                 h_digit, w_digit = cropped_digit.shape
#                 if h_digit > w_digit:
#                     scale = model_input_size[0] / h_digit
#                     w_resized = int(w_digit * scale)
#                     h_resized = model_input_size[0]
#                 else:
#                     scale = model_input_size[1] / w_digit
#                     h_resized = int(h_digit * scale)
#                     w_resized = model_input_size[1]

#                 # Ensure dimensions are at least 1x1 to prevent errors with very small contours
#                 if w_resized == 0: w_resized = 1
#                 if h_resized == 0: h_resized = 1

#                 resized_digit = cv2.resize(cropped_digit, (w_resized, h_resized), interpolation=cv2.INTER_AREA)

#                 # Calculate paste position to center the digit
#                 x_offset = (model_input_size[1] - w_resized) // 2
#                 y_offset = (model_input_size[0] - h_resized) // 2

#                 # Paste the resized digit onto the center of the final image
#                 final_img[y_offset : y_offset + h_resized, x_offset : x_offset + w_resized] = resized_digit

#                 # # --- Debugging Tip: Uncomment to visualize final image sent to model ---
#                 # cv2.imshow("Final Image to Model", final_img)
#                 # cv2.waitKey(0)

#                 # Normalize pixel values to [0, 1]
#                 final_img = 0 + (final_img / 255.0)

#                 # Reshape for the model: Add batch dimension (1) and channel dimension (1 for grayscale)
#                 final_img = final_img.reshape(1, model_input_size[0], model_input_size[1], 1)

#                 ## 2. GET PREDICTION FROM THE MODEL
#                 predictions = model.predict(final_img, verbose=0)
#                 classIndex = np.argmax(predictions, axis=1)
#                 probabilityValue = np.amax(predictions)

#                 ## 3. DECIDE BASED ON CONFIDENCE THRESHOLD
#                 if probabilityValue > confidence_threshold:
#                     result.append(classIndex[0])
#                     digit_found_in_cell = True
        
#         # If no significant contour was found, or filters rejected it, or prediction confidence was low, it's an empty cell
#         if not digit_found_in_cell:
#             result.append(0)

#     return result

#### 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


#### 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver