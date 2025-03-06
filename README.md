# Football Image Processing

This project applies various image processing techniques to football-related images, including edge detection, segmentation, contour detection, and face blurring for privacy.

## Features
- **Image Display & Comparison**: Functions to show images and compare results.
- **Grayscale Conversion**: Converts images to grayscale.
- **Thresholding (Otsu's Method)**: Segments images based on pixel intensity.
- **Edge Detection**: Uses Sobel and Canny filters for edge detection.
- **Gaussian Blur**: Applies blurring effects to images.
- **Exposure Adjustment**: Uses histogram equalization to enhance images.
- **Image Segmentation**: Uses SLIC algorithm for segmentation.
- **Contour Detection**: Detects contours in thresholded images.
- **Corner Detection**: Uses Harris corner detection to find corners in an image.
- **Face Detection**: Detects faces using a trained cascade classifier.
- **Face Blurring**: Blurs detected faces for privacy.

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install numpy matplotlib scikit-image
```

## Usage
1. Place football-related images in the appropriate directory.
2. Update the file paths in the script to match your image locations.
3. Run the script to process and visualize images.
```sh
python Football_Image_Processing.py
```

## Example Functions
- `show_image(image, title)`: Displays an image.
- `compare_images(image1, image2, title1, title2)`: Compares two images.
- `getFaceRectangle(image, d)`: Extracts detected face regions.
- `mergeBlurryFace(original, gaussian_image, d)`: Overlays blurred faces onto the original image.

## Notes
- Ensure the correct file paths for input images.
- Adjust Gaussian blur sigma values for better face blurring results.
- Modify segmentation parameters for different football images.

## License
This project is open-source and free to use.

