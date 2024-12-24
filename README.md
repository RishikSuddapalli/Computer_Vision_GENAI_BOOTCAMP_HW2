# Tags Detection and Segmentation

This project demonstrates the detection and segmentation of tags in images using OpenCV. The pipeline includes thresholding, bounding box detection, and saving the processed images into an organized directory structure.

## Project Objective
To detect tags in an image by:
- Applying binary thresholding for segmentation.
- Detecting bounding boxes around significant regions.
- Saving the processed images, including thresholded images, bounding box overlays, and cropped tag regions.

## Features
- Processes input images to extract tags as separate cropped images.
- Saves intermediate and final results (thresholded image, bounding box overlay, cropped tags) to an output directory.

## How It Works
1. **Load the Image**: Reads the input image from a specified path.
2. **Preprocess the Image**: Converts the image to grayscale and applies binary thresholding using Otsu's method.
3. **Tag Detection**: 
   - Applies morphological closing to remove noise and improve segmentation.
   - Finds contours and filters them based on area to identify tags.
4. **Bounding Box Creation**: Draws bounding boxes around detected tags and merges them for grouped tags.
5. **Cropped Tag Extraction**: Saves cropped images of the detected tags.
6. **Output Storage**: Saves all processed images in an organized output directory.

## Directory Structure
```
project/
├── assignment-2.jpg        # Input image
├── main.py                 # Code for tag detection and segmentation
├── Output/                 # Directory for processed outputs
│   ├── thresholded_image.jpg
│   ├── bounded_image.jpg
│   ├── tag_1.jpg
│   ├── tag_2.jpg
│   └── ...
```

## Technologies and Libraries Used
- **Python** 🐍
- **OpenCV** 👁
- **NumPy** 🔢
- **Matplotlib** 📊

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required libraries:
   ```bash
   pip install opencv-python-headless matplotlib numpy
   ```

## Usage
1. Place the input image (`assignment-2.jpg`) in the project directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. Check the `Output/` directory for the processed images.

## Next Steps
- **Optimization**: Refine tag detection accuracy by enhancing contour filtering techniques.
- **Deep Learning Integration**: Explore advanced methods for tag classification.

## Acknowledgments
Special thanks to Krish Naik and Monal Kumar for their guidance throughout the Computer Vision with Generative AI Bootcamp.
