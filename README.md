# 🧠 Sudoku Solver & OCR Game

A modular Python-based Sudoku application supporting:

* Puzzle generation by difficulty (Easy/Medium/Hard)
* Manual solving with PyGame GUI
* Puzzle solving with selectable algorithms
* Image upload to detect Sudoku using OCR and a trained CNN model

---

## 📁 Project Structure

```
sudoku_ocr/
├── core/
│   ├── puzzle_generator.py      # Generates puzzles and stores them in JSON
│   ├── algorithms/
│   │   ├── backtracking_solver.py
│   │   └── dlx_solver.py        # Different solving algorithms
│   ├── image_to_grid.py         # OCR module: converts image to grid
│   ├── utils.py                 # Shared image processing utilities
│   └── models/
│       └── mnist_model.h5       # Trained CNN digit recognition model
│
├── data/
│   └── puzzles.json             # Latest puzzle and solution
│
├── gui/
│   └── game_ui.py               # Main PyGame UI
│
└── README.md
```

---

## ✅ Prerequisites

* Python 3.7+
* pip
* [Tesseract (optional)](https://github.com/tesseract-ocr/tesseract) for enhanced OCR (currently uses CNN instead)

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/computiphy/sudoku_ocr.git
cd sudoku_ocr

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```txt
pygame
opencv-python
tensorflow
numpy
```

---

## ▶️ Usage

```bash
python main.py
```

This will launch the PyGame interface with a newly generated puzzle.

---

## 🧩 Features

| Feature                | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| Puzzle Generation      | Generates new Sudoku puzzles at easy, medium, or hard levels          |
| Manual Solving UI      | User-friendly grid interface built with PyGame                        |
| Solver Algorithms      | Choose between `backtracking` and `dlx` solving algorithms            |
| Image Upload to Puzzle | Upload an image of a Sudoku and convert it to a playable puzzle (OCR) |
| Cell Navigation        | Navigate grid using arrow keys                                        |
| Check Button           | Validate your current inputs against the solution                     |
| Timer                  | Tracks how long you've been solving                                   |

---

## 🔍 In-Depth Module Descriptions

---

### 🎮 `gui/game_ui.py` — PyGame GUI

Handles:

* Displaying puzzle grid and numbers
* Receiving and processing user input
* Generating new puzzles
* Uploading image for OCR
* Selecting solving algorithm from dropdown
* Solving the puzzle using chosen algorithm

Key Components:

* **Buttons**: "New", "Solve", "Upload", "Algorithm"
* **Keyboard Navigation**: Use arrow keys to move around the grid
* **Check Result**: Press Enter to validate

---

### 🧬 `core/puzzle_generator.py`

Generates a new Sudoku puzzle and its solution using:

* Difficulty level passed as CLI arg (`easy`, `medium`, `hard`)
* Saves puzzle and solution to `data/puzzles.json`

Example:

```bash
python core/puzzle_generator.py easy
```

---

### 📊 `core/algorithms/backtracking_solver.py` & `dlx_solver.py`

Self-contained implementations of different solving techniques. Modular design lets you plug in any new method.

Each must expose:

```python
def solve(grid: List[List[int]]) -> List[List[int]]:
```

---

### 📏 `core/utils.py` — Image Processing Utilities

Contains:

* `preProcess`: Converts image to thresholded grayscale
* `biggestContour`: Finds the largest quadrilateral (assumed puzzle)
* `reorder`: Orders the 4 corners of the grid
* `splitBoxes`: Splits warped image into 9x9 individual boxes
* `getPredection`: Feeds digits into the CNN for classification

---

## 📸 `core/image_to_grid.py` — **OCR Module Deep Dive**

### ✨ Goal:

Extract a Sudoku puzzle from a photo (top-down image), recognize digits using a CNN model, and store the puzzle as JSON.

### Step-by-Step Breakdown:

#### ✅ 1. **Load & Preprocess Image**

```python
img = cv2.imread(image_path)
img = cv2.resize(img, (450, 450))
imgThreshold = preProcess(img)
```

Converts image to grayscale and adaptive thresholded binary for better contour detection.

#### ✅ 2. **Detect Grid Contour**

```python
contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggest, _ = biggestContour(contours)
```

Finds largest quadrilateral contour using OpenCV heuristics.

#### ✅ 3. **Perspective Warp**

```python
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgWarpGray = cv2.warpPerspective(img, matrix, (450, 450))
```

Aligns and flattens the grid to top-down view so each box is uniformly sized.

#### ✅ 4. **Split & Predict Digits**

```python
boxes = splitBoxes(imgWarpGray)
raw_numbers = getPredection(boxes, model)
```

* Crops each of the 81 boxes
* Feeds them to the pretrained CNN (`mnist_model.h5`)
* Applies confidence threshold to ignore noise

#### ✅ 5. **Post-process**

```python
numbers = [n if 1 <= n <= 9 else 0 for n in raw_numbers]
puzzle_grid = [[...]]  # Reformat to 9x9
```

Ensures:

* The predictions form a valid 9x9 matrix
* Unknown digits default to `0`

#### ✅ 6. **Save to JSON**

```python
puzzle_data = {
  "difficulty": "ocr_upload",
  "puzzle": puzzle_grid,
  "solution": puzzle_grid  # Placeholder for manual solving
}
```

Used as the main input for PyGame GUI.

---

## 🔄 Future Enhancements

* Better OCR fallback with Tesseract or LeNet-5
* Web UI using Flask or Streamlit
* Webcam integration for live puzzle detection
* Mobile-friendly port with Kivy

---

## 👨‍💼 Authors

* **Swaroop P** — Design, architecture, GUI, and core engineering
* **OpenAI ChatGPT** — Assistant developer & documentation helper
