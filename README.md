# SEC-DIP-19AI406-License-Plate-Detection-

```
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("car_plate.jpg")

if img is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(blur, 100, 200)

# Find contours
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

# Copy original image
output = img.copy()

detected = False

# Check each contour
for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    # License plates are usually rectangular
    aspect_ratio = w / float(h)

    area = w * h

    # Filter possible license plate regions
    if 2.0 < aspect_ratio < 6.0 and 1000 < area < 100000:

        # Draw rectangle around possible plate
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Extract plate region
        plate = output[y:y+h, x:x+w]

        # Blur the detected plate
        blurred_plate = cv2.GaussianBlur(plate, (25, 25), 0)

        output[y:y+h, x:x+w] = blurred_plate

        detected = True

if detected:
    print("License plate detected.")
else:
    print("No license plate detected. Try changing the parameters.")

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(output_rgb)
plt.title("License Plate Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

```
<img width="1340" height="287" alt="Screenshot 2026-09-16 133230" src="https://github.com/user-attachments/assets/bbc7df86-3e52-4488-bc6d-d0a244dcc958" />



