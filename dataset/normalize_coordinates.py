def normalize(landmarks):
    lms = landmarks
    min_x = 1
    min_y = 1
    max_x = 0
    max_y = 0

    for [x, y, z] in landmarks:
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)

    for lm in lms:
        lm[0] = (lm[0] - min_x) / (max_x - min_x)
        lm[1] = (lm[1] - min_y) / (max_y - min_y)

    return lms