import os
import cv2  # type: ignore
from skimage.filters import prewitt_h, prewitt_v  # type: ignore
from skimage import img_as_float  # type: ignore
import matplotlib.pyplot as plt

# File path
image_path = r'E:\1.MACHINE LEARNING\DATASET\fruit\103_100.jpg'

# Check if the file exists
if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading the image.")
    else:
        print("Image loaded successfully!")

        # 1st SIFT Keypoint Detection
        sift = cv2.SIFT_create()
        keypoints = sift.detect(image, None)

        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #2nd  Canny edge detection
        edges_canny = cv2.Canny(image, 100, 200)

        # Convert the image to float for skimage processing
        image_float = img_as_float(image)

        # 3rd Prewitt edge detection
        edges_prewitt_horizontal = prewitt_h(image_float)
        edges_prewitt_vertical = prewitt_v(image_float)

        # Plot the results
        plt.figure(figsize=(15, 10))

        # Original Image
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        # Canny Edges
        plt.subplot(2, 2, 2)
        plt.title("Canny Edges")
        plt.imshow(edges_canny, cmap='gray')
        plt.axis('off')

        # Prewitt Horizontal Edges
        plt.subplot(2, 2, 3)
        plt.title("Prewitt Horizontal Edges")
        plt.imshow(edges_prewitt_horizontal, cmap='gray')
        plt.axis('off')

        # Prewitt Vertical Edges
        plt.subplot(2, 2, 4)
        plt.title("Prewitt Vertical Edges")
        plt.imshow(edges_prewitt_vertical, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Display the image with SIFT keypoints
        plt.figure(figsize=(10, 5))
        plt.title('SIFT Keypoints')
        plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
