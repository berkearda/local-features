# Local Features Matching Project

This project implements key components of Harris corner detection, descriptor extraction, and matching for local image features. The goal is to detect and match local keypoints between two images using three different techniques for matching descriptors: **One-way Nearest Neighbor**, **Mutual Nearest Neighbor**, and **Ratio Test Matching**.

## Project Structure
```bash
lab02-local-features/
├── functions/              # Directory containing all main functions
│   ├── __init__.py         # Init file for functions module
│   ├── extract_descriptors.py # Keypoint descriptor extraction functions
│   ├── extract_harris.py   # Harris corner detection functions
│   ├── match_descriptors.py # Descriptor matching functions
│   └── vis_utils.py        # Visualization utilities
├── images/                 # Directory for input image files
│   ├── blocks.jpg
│   ├── house.jpg
│   ├── I1.jpg
│   └── I2.jpg
├── .gitignore              # Ignoring unnecessary files and outputs
├── main.py                 # Main script for executing the project
├── open_project.sh         # Shell script to set up the project environment
└── requirements.txt        # List of dependencies for the project
```
## Project Features

This project includes the following key components:

- **Harris Corner Detection**: Detects corner points from images using gradient-based detection.
- **Descriptor Extraction**: Extracts descriptors around the detected keypoints to enable feature comparison.
- **Matching Techniques**:
  - One-way nearest neighbor matching.
  - Mutual nearest neighbor matching.
  - Ratio test matching to reduce false matches.

## Key Components

### Harris Corner Detection
The Harris corner detector is used to identify keypoints in each image. It calculates the image gradient and constructs the auto-correlation matrix to determine corner strength at each pixel.

### Keypoint Descriptor Extraction
Patches are extracted around each keypoint, and descriptors are created to capture local image information for matching.

### Descriptor Matching
Three techniques are implemented to match descriptors between images:

- **One-way Nearest Neighbor**: Matches keypoints in the first image to the nearest keypoint in the second image.
- **Mutual Nearest Neighbor**: Matches keypoints only if they are the nearest neighbors to each other in both images.
- **Ratio Test Matching**: Compares the distance of the nearest match with the second-nearest match, and retains only matches that pass a set threshold.

## Results and Visualizations

The following techniques are implemented and visualized for matching descriptors:

1. **One-way Nearest Neighbor Matching**
2. **Mutual Nearest Neighbor Matching**
3. **Ratio Test Matching**

## Installation and Setup

### Requirements

- Python 3.8+
- Install all dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
## How to Run

1. Clone the repository:
```bash
git clone https://github.com/berkearda/local-features.git
cd local-features
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the main script:
```bash
python main.py
```
