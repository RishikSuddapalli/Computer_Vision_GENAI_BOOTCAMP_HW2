import cv2
import numpy as np
import os

def process_and_save_image(image_path, output_dir='Output'):
    """
    This function processes an image, applies thresholding, detects bounding boxes for tags,
    and saves images including the thresholded image, the image with bounding boxes, and 
    cropped images of the detected tags into an output directory.
    
    Parameters:
        image_path (str): Path to the input image that needs to be processed.
        output_dir (str): Directory where the processed images will be saved. Default is 'Output'.
    """
    # Load the input image
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError("Error: Image not found at the specified path!")
    print("Image loaded successfully.")

    # Convert the image to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to separate the background and tags
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the thresholded image
    cv2.imwrite(os.path.join(output_dir, 'thresholded_image.jpg'), thresh)
    print(f"Thresholded image saved to {output_dir}/thresholded_image.jpg")

    # Perform morphological closing to remove noise and fill gaps in the thresholded image
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    # Find contours in the closed image to identify potential areas of interest (tags)
    filtered_contours = [ctr for ctr in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(ctr) > 5000]

    # Prepare for bounding box drawing and cropping the tags
    image_result = image.copy()  # Copy the original image to draw bounding boxes
    all_tags = []  # List to store coordinates of detected bounding boxes
    cropped_images = []  # To store cropped images of the detected tags

    # Loop through each filtered contour to find and crop the tags
    for row_contour in filtered_contours:
        x_row, y_row, w_row, h_row = cv2.boundingRect(row_contour)
        row_boxes = []

        # Loop through each third of the bounding box to detect individual tags
        for j in range(3):
            tag_x = max(0, x_row + j * (w_row // 3) - 8)  # Add margin on the left
            tag_y = max(0, y_row - 15)  # Add margin on the top
            tag_w = (w_row // 3) + 16  # Add margin on both sides
            tag_h = h_row + 47  # Add margin to the height

            if tag_x + tag_w > image.shape[1]:
                tag_w = image.shape[1] - tag_x  # Ensure the tag width doesn't exceed image boundaries

            row_boxes.append((tag_x, tag_y, tag_w, tag_h))  # Add the current tag's box coordinates

            # Once all 3 tags are detected, merge them into one bounding box
            if len(row_boxes) == 3:
                min_x = min(box[0] for box in row_boxes)
                min_y = min(box[1] for box in row_boxes)
                max_x = max(box[0] + box[2] for box in row_boxes)
                max_y = max(box[1] + box[3] for box in row_boxes)

                # Draw the merged bounding box on the image
                cv2.rectangle(image_result, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Green bounding box

                # Crop the detected tag from the image using the bounding box coordinates
                tag_crop = image[min_y:max_y, min_x:max_x]
                cropped_images.append(tag_crop)  # Store the cropped tag image

                # Save the cropped tag as a separate image file
                cv2.imwrite(os.path.join(output_dir, f'tag_{len(all_tags) + 1}.jpg'), tag_crop)
                print(f"Cropped image saved to {output_dir}/tag_{len(all_tags) + 1}.jpg")

                all_tags.append((min_x, min_y, max_x - min_x, max_y - min_y))  # Store the tag's coordinates
                row_boxes = []  # Clear the temporary list for the next set of tags

    # Save the final image with bounding boxes drawn
    cv2.imwrite(os.path.join(output_dir, 'bounded_image.jpg'), image_result)
    print(f"Image with bounding boxes saved to {output_dir}/bounded_image.jpg")

    # Display the result with bounding boxes for visual confirmation using OpenCV
    cv2.imshow("Final Tags with Merged Bounding Boxes", image_result)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    print(f"Total tags detected: {len(all_tags)}")

# Example usage
image_path = 'assignment-2.jpg'
process_and_save_image(image_path)
