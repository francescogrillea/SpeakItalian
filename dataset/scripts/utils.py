import cv2

def normalize(landmarks):
    # Normalize x and y w.r.t. their min and max values
    lms = landmarks
    min_x = 1
    min_y = 1
    max_x = 0
    max_y = 0

    for [x, y, _] in landmarks:
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)

    for lm in lms:
        lm[0] = (lm[0] - min_x) / (max_x - min_x)
        lm[1] = (lm[1] - min_y) / (max_y - min_y)

    return lms


def create_dict(landmarks, label):
    # Create a dictionary from the landmarks in such a way to create the dataframe easily
    dictionary = {}
    for enu, [x, y, z] in enumerate(landmarks):
        dictionary[landmarks_name[enu] + "_x"] = x
        dictionary[landmarks_name[enu] + "_y"] = y
        dictionary[landmarks_name[enu] + "_z"] = z

    dictionary["class"] = label
    return dictionary


landmarks_name = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMBJP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP"
}


labels = [
    "thumbUp",
    "thumbDown",
    "okay",
    "gun",
    "call",
    "fist",
    "stop",
    "peace",
    "rock",
    "index"
]

def show_text(image, text, x=10, y=50):
    """Show 'text' onto 'img' at (x, y)
    """

    # Add some fading by showing the text in black translated
    cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
