import numpy as np
import cv2


def capture_motion():
    cap = cv2.VideoCapture(0)
    last_frame = None
    while True:
        _, raw = cap.read()
        # cv2.imshow("raw", raw)
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("gray", gray)
        if last_frame is None:
            last_frame = gray
            continue
        diff = np.array(np.abs(np.array(gray, dtype=np.int16) -
                        np.array(last_frame, dtype=np.int16)), dtype=np.uint8)
        # cv2.imshow("diff", diff)
        diff = cv2.GaussianBlur(diff, (5, 5), 1.2)
        _, threshold = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        # cv2.imshow("threshold", threshold)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        image_open = cv2.morphologyEx(
            threshold, cv2.MORPH_OPEN, kernel=kernel_open)
        # cv2.imshow("image_open", image_open)
        image_close = cv2.morphologyEx(
            image_open, cv2.MORPH_CLOSE, kernel=kernel_close)
        # cv2.imshow("image", image_close)
        image_open_clone = image_open.copy()
        edges_blur = cv2.blur(image_open_clone, (3, 3))
        # edges_blur = image_open_clone.copy()
        # print(edges_blur)
        edges = cv2.Canny(edges_blur, 50, 255, apertureSize=3)
        cv2.imshow('edges_blur', edges_blur)
        cv2.imshow('edges', edges)
        lines = cv2.HoughLines(edges, 1, np.pi/90, 110)
        if lines is None:
            lines = []
        else:
            print(f"lines is not None")

        result2 = np.zeros(gray.shape, dtype=gray.dtype)

        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(result2, (x1, y1), (x2, y2), 255, 2)
        cv2.imshow("result2", result2)

        result3 = np.zeros(gray.shape, dtype=gray.dtype)
        lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 50)
        if lines2 is None:
            lines2 = []
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(result3, (x1, y1), (x2, y2), 255, 2)
        cv2.imshow("result3", result3)

        # d = cv2.findContours(image_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(d)
        contours, hierarchy = cv2.findContours(
            image_open_clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # image_open_clone_ = cv2.drawContours(
        #     image_open_clone.copy(), contours, -1, 127, 3)
        # cv2.imshow("box", image_open_clone_)
        # result = np.zeros(gray.shape, dtype=gray.dtype)
        # for i in range(0, len(contours)):
        #     x, y, w, h = cv2.boundingRect(contours[i])
        #     cv2.rectangle(result, (x, y), (x+w, y+h), 127, 5)
        # cv2.imshow("result", result)

        # found_max = None
        # found_max_box = [0, 0]
        # for i in range(0, len(contours)):
        #     x, y, w, h = cv2.boundingRect(contours[i])
        #     if w > found_max_box[0] and h > found_max_box[1]:
        #         found_max = contours[i]
        #         found_max_box = [w, h]
        # if found_max is None:
        #     continue
        # result = np.zeros(gray.shape, dtype=gray.dtype)
        # result = cv2.drawContours(result, [found_max, ], -1, 127, 3)
        # cv2.imshow("result", result)

        # cv2.rectangle(result, (x, y), (x+w, y+h), 127, 5)
        # print(found_max)

        bounding_rects = [(c, cv2.boundingRect(c)) for c in contours]
        list.sort(bounding_rects, key=lambda x: x[1][2]+x[1][3], reverse=True)
        result = np.zeros(gray.shape, dtype=gray.dtype)
        result = cv2.drawContours(
            result, [x[0] for x in bounding_rects[:2]], -1, 255, 3)
        result4 = np.zeros(gray.shape, dtype=gray.dtype)
        outline = None
        if len(bounding_rects) >= 2:
            outline = [min(bounding_rects[0][1][0], bounding_rects[1][1][0]), min(bounding_rects[0][1][1], bounding_rects[1][1][1]),
                    max(bounding_rects[0][1][2], bounding_rects[1][1][2]), max(bounding_rects[0][1][3], bounding_rects[1][1][3])]
            for x in bounding_rects[:2]:
                x, y, w, h = x[1]
                result4 = cv2.rectangle(result4, (x, y), (x+w, y+h), 127, 5)
        else:
            if len(bounding_rects) == 1:
                outline = bounding_rects[0][1]
        if outline is None:
            continue

        result4 = cv2.rectangle(result4, tuple(outline[0:2]), tuple(np.array(outline[0:2]) + np.array(outline[2:4])), 127, 5)
        
        outline_box = [*outline[0:2], *(np.array(outline[0:2]) + np.array(outline[2:4]))]

        plot = gray.copy()
        center = tuple(map(int, [(outline_box[0] + outline_box[2]) / 2, (outline_box[1] + outline_box[3]) / 2]))
        pts = [
            ((outline_box[0] + outline_box[2]) / 2, outline_box[1]),
            ((outline_box[0] + outline_box[2]) / 2, outline_box[3])
        ]
        cv2.circle(result4, center, 5, 255, -1)
        for p in pts:
            cv2.circle(result4, tuple(map(int, p)), 3, 255, -1)

        last_frame = gray
        cv2.imshow("result4", result4)
        cv2.imshow("result", result)
        k = chr(cv2.waitKey(1) & 0xFF)
        if k == 'q':
            break


if __name__ == '__main__':
    capture_motion()
