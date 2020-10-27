import numpy as np
import cv2
import json
from subprocess import call
import sys

argv1 = sys.argv[1]  # get camera number as terminal argument


#       --> read JSON file "data" <--
try:
    with open('data.json') as f:
        data = json.load(f)

    p = data['points']
    for i in range(4):
        p[i] = (p[i]['x'], p[i]['y'])

    K = data['K']

except:
    print("JSON file doesn't exist yet")

    data = {}

    data['colors'] = {}

    p = []
    for i in range(4):
        ponto = ()
        p.append(ponto)

    # set points as corners of the cam img
    p[0] = (89, 2)
    p[1] = (597, 30)
    p[2] = (567, 480)
    p[3] = (55, 455)

    K = 15

try:
    with open("data.json") as f:
        data = json.load(f)
    cam_parameters = data['camera_parameters']

except:
    cam_parameters = f"v4l2-ctl -d /dev/video{argv1} -c saturation=255 -c gain=255 -c exposure_auto=1 -c exposure_absolute=40 -c focus_auto=0"


#       --> set camera parameters <--
call(cam_parameters.split())


#       --> border calibration <--
def transform(img, p0, p1, p2, p3):
    # array containing the four corners of the field
    inputQuad = np.array([p0, p1, p2, p3],  dtype="float32")
    outputQuad = np.array([(0, 0),  # array containing the four corners of the image
                           (450-1, 0),
                           (450-1, 390-1),
                           (0, 390-1)], dtype="float32")

    # Get the Perspective Transform Matrix i.e. lambda
    lbd = cv2.getPerspectiveTransform(inputQuad, outputQuad)

    # Apply the Perspective Transform just found to the src image
    output = cv2.warpPerspective(img, lbd, (450, 390))

    return output


cap = cv2.VideoCapture(int(argv1))

if not cap.isOpened():
    print("Can't open the video cam")
    quit()

cap.set(3, 1280)
cap.set(4, 720)
dWidth = int(cap.get(3))  # get the width of frames of the video
dHeight = int(cap.get(4))  # get the height of frames of the video
print("Frame size:", dWidth, "x", dHeight)  # print image size

cv2.namedWindow("Control", cv2.WINDOW_AUTOSIZE)


def callback0(val):
    p[0] = (val, p[0][1])


def callback1(val):
    p[0] = (p[0][0], val)


def callback2(val):
    p[1] = (val, p[1][1])


def callback3(val):
    p[1] = (p[1][0], val)


def callback4(val):
    p[2] = (val, p[2][1])


def callback5(val):
    p[2] = (p[2][0], val)


def callback6(val):
    p[3] = (val, p[3][1])


def callback7(val):
    p[3] = (p[3][0], val)


cv2.createTrackbar("P0-X", "Control", p[0][0], dWidth, callback0)
cv2.createTrackbar("P0-Y", "Control", p[0][1], dHeight, callback1)
cv2.createTrackbar("P1-X", "Control", p[1][0], dWidth, callback2)
cv2.createTrackbar("P1-Y", "Control", p[1][1], dHeight, callback3)
cv2.createTrackbar("P2-X", "Control", p[2][0], dWidth, callback4)
cv2.createTrackbar("P2-Y", "Control", p[2][1], dHeight, callback5)
cv2.createTrackbar("P3-X", "Control", p[3][0], dWidth, callback6)
cv2.createTrackbar("P3-Y", "Control", p[3][1], dHeight, callback7)

# Optimal values
# P0 : (111,47) P1 : (566,10) P2 : (579, 415) P3 : (135, 421)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (450, 390))
    transformed_frame = transform(frame, p[0], p[1], p[2], p[3])

    transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2LAB)

    cv2.circle(frame, p[0], 5, (255, 0, 0), -1)
    cv2.circle(frame, p[1], 5, (0, 255, 0), -1)
    cv2.circle(frame, p[2], 5, (0, 0, 255), -1)
    cv2.circle(frame, p[3], 5, (255, 255, 255), -1)

    # Display the resulting frame
    cv2.imshow('MyVideo_Original', frame)
    cv2.imshow("MyVideo_Transformed", transformed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

transformed_frame = cv2.fastNlMeansDenoisingColored(
    transformed_frame, None, 3, 3, 7, 21)


#       --> color calibration <--
img = transformed_frame
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-4)

while(True):
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # create numbered color rectangles of each cluster
    retangulos = np.zeros((200, 1200, 3), np.uint8)
    rect_size = 1200 // K
    for i in range(K):
        # color_rect = tuple([int(x) for x in center[i]])
        color_rect = np.uint8([[[int(x) for x in center[i]]]])
        color_rect = cv2.cvtColor(color_rect, cv2.COLOR_LAB2BGR)
        color_rect = tuple([int(x) for x in np.reshape(color_rect, (-1))])
        print(color_rect)

        cv2.rectangle(retangulos, (i*rect_size, 0),
                      ((i+1)*rect_size, 150), color_rect, thickness=-1)
        cv2.putText(retangulos, str(i), (i*rect_size + rect_size//2 - 15, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=3)

    # display original image, clustered image and color rectangles

    rgb_img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    rgb_clusterized = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
    cv2.imshow('img', rgb_img)
    cv2.imshow('clusterized_img', rgb_clusterized)
    cv2.imshow('colors', retangulos)
    cv2.waitKey(30)
    cv2.imshow('img', rgb_img)
    cv2.imshow('clusterized_img', rgb_clusterized)
    cv2.imshow('colors', retangulos)
    cv2.waitKey(30)

    # get value for K
    tmp = input('Insert number of clusters: ')
    if tmp == 'q':
        break
    else:
        K = int(tmp)

color_list = []
label = list(label.flatten())
max_color_values = []
min_color_values = []
recognizable_color_names = ['blue', 'yellow', 'orange', 'salmon',
                            'pink', 'green', 'purple', 'red', 'brown', '']

for i in range(K):
    color_name = input('Cluster {}: '.format(i))
    while color_name not in recognizable_color_names:
        print("Unrecognizable color.")
        color_name = input('Cluster {}: '.format(i))

    if color_name != '':
        color_list.append([color_name, tuple([int(x) for x in center[i]]), i])

        # get max and min BGR values for each named color
        x = [Z[j] for j in range(len(label)) if label[j] == i]
        x = np.array(x)
        max_color_values.append(
            (np.max(x[:, 0]), np.max(x[:, 1]), np.max(x[:, 2])))
        min_color_values.append(
            (np.min(x[:, 0]), np.min(x[:, 1]), np.min(x[:, 2])))


#       --> write to JSON file "data" <--
color_dict = {}

for i in range(len(color_list)):
    color_list[i][1] = {'B_max': int(max_color_values[i][0]), 'B_min': int(min_color_values[i][0]),
                        'G_max': int(max_color_values[i][1]), 'G_min': int(min_color_values[i][1]),
                        'R_max': int(max_color_values[i][2]), 'R_min': int(min_color_values[i][2])}
    color_dict[color_list[i][0]] = color_list[i][1]

for i in range(4):
    p[i] = {'x': p[i][0], 'y': p[i][1]}

for color in color_dict:
    data['colors'][color] = color_dict[color]
data['points'] = p
data['K'] = K
data['camera_parameters'] = cam_parameters

with open('data.json', 'w') as f:
    json.dump(data, f, indent=True, ensure_ascii=False)
