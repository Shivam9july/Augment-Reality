import cv2
import numpy as np

MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=5000)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def load_input(image_path):
    input_image = cv2.imread(image_path)

    input_image = cv2.resize(input_image, (300, 400), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return gray_image, keypoints, descriptors

def compute_matches(descriptors_input, descriptors_output):
    if len(descriptors_output) != 0 and len(descriptors_input) != 0:
        matches = flann.knnMatch(np.asarray(descriptors_input, np.float32), np.asarray(descriptors_output, np.float32),
                                 k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.69 * n.distance:
                good.append(m)
        return good
    else:
        return None

def draw_transparent_box_with_text(frame, x, y, w, h, text):
    overlay = frame.copy()

    # Increase the width of the transparent rectangle
    new_w = w + 100  # Adjust the width as needed

    # Draw transparent rectangle on the overlay with light blue color
    cv2.rectangle(overlay, (x, y), (x + new_w, y + h), (173, 216, 230, 100), -1)

    # Add text inside the transparent box
    font_size = 0.4
    line_spacing = 20
    text_lines = text.split("\n")
    text_position_y = y + h // 2 - (len(text_lines) * line_spacing) // 2

    for line in text_lines:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        text_position = (x + new_w // 2 - text_size[0] // 2, text_position_y)
        cv2.putText(overlay, line, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255, 100), 2)
        text_position_y += line_spacing

    # Blend the overlay with the frame
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

if __name__ == '__main__':
    # Load information for multiple books
    book_info = [
        {"image_path": "copy.jpg", "text": "chitransh ki copy"},
        {"image_path": "kunal.jpg", "text": "kunal messy wala "},
        
        
    ]

    input_data = []
    for book in book_info:
        input_data.append(load_input(book["image_path"]))

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()
        all_matches = []
        for idx, book_data in enumerate(input_data):
            if len(book_data[1]) < MIN_MATCHES:
                continue

            frame = cv2.resize(frame, (600, 450))
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)

            matches = compute_matches(book_data[2], output_descriptors)
            all_matches.append((matches, idx))

        if all_matches:
            matches, selected_book_idx = max(all_matches, key=lambda x: len(x[0]))
            if len(matches) > 10:
                src_pts = np.float32([input_data[selected_book_idx][1][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([output_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                pts = np.float32([[0, 0], [0, 399], [299, 399], [299, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw transparent box with longer text inside on the right side of the book (light blue color)
                x, y, w, h = cv2.boundingRect(dst.astype(int))
                text = book_info[selected_book_idx]["text"]
                draw_transparent_box_with_text(frame, x + w, y, 50, h, text)

                cv2.imshow('Final Output', frame)
            else:
                cv2.imshow('Final Output', frame)
        else:
            cv2.imshow('Final Output', frame)

        key = cv2.waitKey(15)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
