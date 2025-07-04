import cv2
import numpy as np
import os
import json
from core.utils import preProcess, biggestContour, reorder, splitBoxes, getPredection


def extract_grid_from_image(image_path: str):
    heightImg, widthImg = 450, 450
    model = getPretrainedModel()  # Ensure your CNN model loading logic is here

    # Step 1: Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = cv2.resize(img, (widthImg, heightImg))
    imgThreshold = preProcess(img)

    # Step 2: Detect contours
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, max_area = biggestContour(contours)


    if biggest.size == 0:
        raise ValueError("No Sudoku grid found in the image.")

    # Step 3: Warp the perspective to a square
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpGray = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgWarpGray = cv2.cvtColor(imgWarpGray, cv2.COLOR_BGR2GRAY)

    # Step 4: Split image to 81 boxes and predict digits
    boxes = splitBoxes(imgWarpGray)
    raw_numbers = getPredection(boxes, model)
    numbers = []
    for num in raw_numbers:
        try:
            n = int(num)
            numbers.append(n if 1 <= n <= 9 else 0)
        except:
            numbers.append(0)

    if len(numbers) != 81:
        numbers = [0] * 81

    puzzle_grid = [[int(cell) for cell in row] for row in np.array_split(np.array(numbers), 9)]

    # Step 5: Save puzzle to puzzles.json
    puzzle_data = {
        "difficulty": "ocr_upload",
        "puzzle": puzzle_grid,
        "solution": puzzle_grid  # Placeholder for now
    }
    print(puzzle_data)
    with open("data/puzzles.json", "w") as f:
        json.dump(puzzle_data, f)

    return puzzle_grid


def getPretrainedModel():
    # Placeholder: change this to your actual CNN model loader
    from tensorflow.keras.models import load_model
    return load_model("core/models/mnist_printed_model3.h5")
