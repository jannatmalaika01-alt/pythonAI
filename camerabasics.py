import cv2
import numpy as np
import json
import os
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# GEMINI SETUP
# ============================================================

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def extract_text_from_image(image_path):
    """Sends the scanned image to Gemini and gets back the text it reads."""
    img = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            "Transcribe all readable text from this image exactly as written. "
            "Do not summarize or add commentary.",
            img
        ]
    )
    return response.text


def generate_flashcards(text, count=5):
    """Sends the text to Gemini and gets back flashcards as JSON."""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=(
            f"Create exactly {count} flashcards from this text. "
            f'Return a JSON array like: [{{"question": "...", "answer": "..."}}]'
            f"\n\nTEXT:\n{text}"
        ),
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)


# ============================================================
# BOUNDARY DETECTION HELPERS
# ============================================================

def order_points(points):
    rectangle = np.zeros((4, 2), dtype="float32")
    total = points.sum(axis=1)
    rectangle[0] = points[np.argmin(total)]      # top-left
    rectangle[2] = points[np.argmax(total)]      # bottom-right
    difference = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(difference)] # top-right
    rectangle[3] = points[np.argmax(difference)] # bottom-left
    return rectangle


def four_point_transform(image, points):
    rectangle = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rectangle

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    max_height = max(int(height_left), int(height_right))

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rectangle, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


# ============================================================
# MAIN LOOP
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    display = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close small gaps in the edges — Canny often leaves the
    # page outline slightly broken, which stops findContours from seeing
    # it as one closed shape.
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    document = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            document = approx
            break

    scanned = None

    if document is not None:
        cv2.drawContours(display, [document], -1, (0, 255, 0), 3)
        points = document.reshape(4, 2)
        scanned = four_point_transform(frame, points)
        cv2.imshow("Scanned Document", scanned)
    else:
        # Fallback: use the biggest contour's rotated bounding box if no
        # clean 4-corner shape was found, so you still see something.
        if contours and cv2.contourArea(contours[0]) > 5000:
            rect = cv2.minAreaRect(contours[0])
            box = np.array(cv2.boxPoints(rect), dtype="float32")
            cv2.drawContours(display, [box.astype(int)], -1, (0, 165, 255), 2)
            scanned = four_point_transform(frame, box)
            cv2.imshow("Scanned Document (fallback)", scanned)

        cv2.putText(display, "No clean rectangle found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Document Scanner", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and scanned is not None:
        cv2.imwrite("scanned_document.jpg", scanned)
        print("Scanned document saved! Reading it now...")

        text = extract_text_from_image("scanned_document.jpg")
        print("\n--- TEXT FOUND ---\n", text)

        cards = generate_flashcards(text)
        print("\n--- FLASHCARDS ---")
        for i, card in enumerate(cards, 1):
            print(f"{i}. Q: {card['question']}")
            print(f"   A: {card['answer']}")

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()