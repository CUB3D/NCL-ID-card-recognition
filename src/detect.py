import cv2
import numpy as np
import pytesseract
import re
from collections import namedtuple
from matplotlib import pyplot as plt
import os

keypointData = namedtuple("keypointData", "kp1 d1 kp2 d2")
processedImage = namedtuple("processedImage", "original greyscale size")
pair = namedtuple("pair", "x y")
results = namedtuple("results", "studentID libraryID success")

FLANN_CONFIG = {
    "algorithm": 1,
    "trees": 5
}

# Set to true to show intermediate steps
DEBUG = False


def debug_show(img):
    """
    Display an image if debugging is enabled
    :param img: The image to display
    """
    if not DEBUG:
        return
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def get_key_points(img):
    orb = cv2.ORB_create(edgeThreshold=10, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2,
                         nfeatures=1000000)
    return orb.detectAndCompute(img, None)


def get_preprocessed_image(image):
    if image is None:
        raise Exception("Image is invalid")
    return processedImage(image, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), image.shape[:2])


def create_key_point_pair(src):
    kp1, d1 = get_key_points(src.greyscale)
    kp2, d2 = get_key_points(template.greyscale)

    print("Got {} and {} key points".format(len(d1), len(d2)))

    return keypointData(kp1, d1, kp2, d2)


def do_FLANN_match(keypoint_data):
    """
    Perform FLANN knnMatch on a pair of key points
    :param keypoint_data: The pair of key points
    :return: The good matches
    """
    flann = cv2.FlannBasedMatcher(FLANN_CONFIG, {
        "checks": 50
    })

    matches = flann.knnMatch(np.asarray(keypoint_data.d2, np.float32), np.asarray(keypoint_data.d1, np.float32), k=2)

    return [p.x for p in filter(lambda p: p.x.distance < 0.7 * p.y.distance, [pair(m, n) for m, n in matches])]


def get_image_bounding_body(image):
    h, w = image.size
    return np.float32([[0, 0],
                       [0, h - 1],
                       [w - 1, h - 1],
                       [w - 1, 0]]).reshape(-1, 1, 2)


# TODO: rename to extract object
def detect_perspectiveTransform(key_point_data, good_matches, src):
    """
    Generate the perspective transform for extracting
    :param key_point_data:
    :param good_matches:
    :param src:
    :return:
    """
    # Get all the keypoints that were good matches between the template and the src
    src_pts = np.float32([key_point_data.kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_point_data.kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    perspective_correction_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(template.original, key_point_data.kp2, src.original, key_point_data.kp1, good_matches, None,
                           **draw_params)
    debug_show(img3)

    h, w = template.size
    pts = get_image_bounding_body(template)

    # get the perspective correction transform
    dst = cv2.perspectiveTransform(pts, perspective_correction_matrix)

    img3 = cv2.polylines(src.greyscale, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    debug_show(img3)

    dst = cv2.perspectiveTransform(pts, perspective_correction_matrix)
    perM = cv2.getPerspectiveTransform(np.float32(dst), pts)
    fnd = cv2.warpPerspective(src.original, perM, (w, h))

    debug_show(fnd)

    return fnd


def first(lst):
    return next(iter(lst), None)


def extract_text(img):
    info = img[400:1200, 600:1600]
    debug_show(info)

    info = cv2.cvtColor(info, cv2.COLOR_BGR2GRAY)
    info = cv2.threshold(info, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(info)

    print("Found full text:\n---")
    print(text)
    print("---")

    student_id = first(re.findall("[0-9]{9}", text))
    library_id = first(re.findall("[A-Z][0-9]{7}[A-Z]", text))

    # Currently we don't care about anything other than the student id
    return results(student_id, library_id, (student_id is not None))


template = get_preprocessed_image(cv2.imread(os.path.join(os.getcwd(), "static/card_blank.jpg")))


def image_from_bytes(data):
    return cv2.imdecode(np.array(bytearray(data), dtype=np.uint8), -1)


def extract_card_info(image_data):
    processed_image = get_preprocessed_image(image_data)

    key_points = create_key_point_pair(processed_image)
    good_matches = do_FLANN_match(key_points)

    if len(good_matches) > 10:
        image_mapped_to_template = detect_perspectiveTransform(key_points, good_matches, processed_image)
        results = extract_text(image_mapped_to_template)

        if results.success:
            return {
                "Status": 0,
                "StudentID": results.studentID,
                "LibraryID": results.libraryID
            }

    return {
        "Status": 1,
        "Error": "ID card not found in image"
    }
