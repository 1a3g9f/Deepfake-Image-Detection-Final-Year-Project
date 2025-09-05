from PIL import Image, ImageChops, ImageEnhance
import urllib.request
import sys, os.path
import numpy as np
import math
import cv2


def detect_lines_and_count_pixels(binary_image_path):
    # Load the binary image
    image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path {binary_image_path} not found.")

    # Detect edges using Canny (optional but can help with better line detection)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        return []

    line_pixel_counts = []
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Create a mask to draw the line
        line_mask = np.zeros_like(image)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

        # Count the number of pixels in the line
        line_pixel_count = cv2.countNonZero(cv2.bitwise_and(image, line_mask))
        line_pixel_counts.append(line_pixel_count)

    return line_pixel_counts

# Apply Error Level Analysis
def ErrorLevelAnalysis(ImageFile, Type):

    # TYPE 1 (IMAGE)
    # TYPE 2 (URL)
    # Load Original Image
    
    TEMP = "J:\Important\C. SC\Flask\static\Images\Image1.jpg"
    Filename = "J:\Important\C. SC\Flask\static\Images\Image.jpg"
    quality = 85
    threshold = 150

    OriginalImage = None

    if Type == 1:
        OriginalImage = Image.open(ImageFile)
    elif Type == 2:
        # Download the file
        urllib.request.urlretrieve(ImageFile, Filename)
        OriginalImage = Image.open(Filename)

    if OriginalImage is None:
        return "Couldn't process your request."

    # Convert to Black and White
    OriginalImage = OriginalImage.convert('L')
    OriginalImage = ImageEnhance.Sharpness(OriginalImage).enhance(0.0)
    # print('Image: %s' % (Filename))

    # Compress at 70%
    OriginalImage.save(TEMP, 'JPEG', quality = quality)
    Temporary = Image.open(TEMP)

    # Find Difference between original and temporary
    Difference = ImageChops.difference(OriginalImage, Temporary)

    # Find the max value of color band in image
    Extrema = Difference.getextrema()
    MaximumDifference = Extrema[1]
    # print('Extrema: %d' % (MaximumDifference), end = ' ')
    Scale = 255.0 / MaximumDifference

    # Enhance the image based on that scale
    Difference = ImageEnhance.Brightness(Difference).enhance(Scale)

    # Fetch the histogram of the difference image (Count of color pixels)
    Lists = Difference.histogram(mask = None, extrema = None)

    # Calculate Threshold by keeping last 75 pixels
    Pixels = 0
    Width, Height = OriginalImage.size
    for i in range(255, 1, -1):
        # Change
        if Pixels + Lists[i] <= (Height * Width / 1200):
            Pixels += Lists[i]
        else:
            Threshold = i + 1
            # print('Threshold: %d' % (Threshold), end = ' ')
            break

    # Apply Threshold
    BlackWhite = Difference.point(lambda x : 0 if x < Threshold else 255, '1')
    BlackWhite.save(TEMP, 'JPEG', quality = quality)
    # Calculate Radius
    WIDTH, HEIGHT = BlackWhite.size
    RADIUS = int(((WIDTH + HEIGHT) - (math.sqrt((WIDTH * HEIGHT)))) / 2)
    print('Radius: %d' % (RADIUS))

    # Maintain a pixel array (Pixel Number : X-Y Co-ordinate)
    Coordinates = []

    # EDGES {(V1->V2) (V2->V3) (V4->V5)}
    Edges = []

    # Scan the entire image and fetch co-ordinates of white pixels
    BlackWhiteImage = BlackWhite.load()
    for x in range(WIDTH):
        for y in range(HEIGHT):

            # Fetch each pixel
            PixelColor = BlackWhiteImage[x, y]

            # If pixel is white, record its co-ordinates
            if PixelColor == 255:
                Coordinates.append([x, y])

    # Loop through XY Co-Ordinates and find Edges
    for coord in Coordinates:

        Index = Coordinates.index(coord)

        x1 = coord[0]
        y1 = coord[1]

        for next_index in range(Index + 1, len(Coordinates)):

            x2 = Coordinates[next_index][0]
            y2 = Coordinates[next_index][1]

            Distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

            # Change n
            if (Distance < 2):
                Edges.append([Index, next_index])

    # Create a list that has connections for every pixel (V1:V2,V3,...) (V2:V1,V2,...)

    # No of white pixels
    TotalWhitePixels = len(Coordinates)
    ConnectedPixelsList = getConnectedPixels(Edges, TotalWhitePixels)

    # Labels of clusters (Starting value -> 100)
    LabelCount = 100

    # Dictionary (Pixel : Label)
    LabelOfPixels = {}

    # Assign every pixel a label of 0
    for index in range(0, len(ConnectedPixelsList)):
        LabelOfPixels.update({ConnectedPixelsList[index][0]: 0})

    # Check neighbor of every pixel and find the root label of every pixel
    for index in range(0, len(ConnectedPixelsList)):
        Element = ConnectedPixelsList[index]

        # Arbitrary root number
        Root = 1000

        # Find label with lowest value and assign to root
        for i in range(1, len(Element)):
            Pixel = Element[i]
            Label = LabelOfPixels.get(Pixel)

            if Label > 0 and Root > Label:
                Root = Label

        # Union-Find Algorithm #

        # If no root found, assign an arbitrary label #
        if Root == 1000:
            LabelCount += 1
            LabelOfPixels.update({Element[0]: LabelCount})
        else:
            # Update all pixels with root as label #
            for i in range(0, len(Element)):
                Pixel = Element[i]
                LabelOfPixels.update({Pixel: Root})

    # # Count the number of white pixels for each label #
    ColorCount = {}
    for lp in LabelOfPixels.items():
        Label = lp[1]
        if Label not in ColorCount:
            ColorCount.update({Label: 1})
        else:
            ColorCount.update({Label: ColorCount.get(Label) + 1})

    # White Pixels in each Cluster #
    # print('White pixels in each cluster: ', end = ' ')

    max2 = 0
    cluster_values = []
    for p in ColorCount.values():
        cluster_values.append(p)
        if max2 < p:
            max2 = p
        print(p, end='  ')
    print()
    print('Max: %d' % (max2))

    # Change
    cluster_values_sorted = sorted(cluster_values)
    squared_lst = [x ** 2 for x in cluster_values_sorted]

    l = len(cluster_values_sorted)
    # sl = int(l * 3 / 4) - 1
    x2 = l // 2 - 1
    ans = sum(squared_lst[x2 : ]) + max2

    maxToTotalRatio = (max2 ** 2 / TotalWhitePixels) * 100
    meanToTotalRatio = (ans / TotalWhitePixels) * 100
    product = meanToTotalRatio * maxToTotalRatio

    return product
    

# Find connections for each pixel #
def getConnectedPixels(edges, total_white_pixels):
    cpl = []

    for index in range(0, total_white_pixels):

        # Create new element for new pixel #
        cpl.append([index])
        # Index of the last element of list #
        cpl_index = (len(cpl) - 1)

        for i in range(0, len(edges)):

            edge = edges[i]
            start = edge[0]
            end = edge[1]

            # Edges (V1->V2) where {start: V1, end: V2} #
            if start == index:
                cpl[cpl_index].append(end)
                continue
            elif end == index:
                cpl[cpl_index].append(start)
                continue

        # Remove all pixels with no connections #
        if len(cpl[cpl_index]) == 1:
            cpl.remove(cpl[cpl_index])

    return cpl


def EdgeDetection(Filename):

    TEMP = "J:\Important\C. SC\Flask\static\Images\Image1.jpg"
    quality = 85

    ImageFile = Filename
    OriginalImage = Image.open(ImageFile)

    # Convert to Black and White
    OriginalImage = OriginalImage.convert('L')
    OriginalImage = ImageEnhance.Sharpness(OriginalImage).enhance(0.0)

    # Compress at 70%
    OriginalImage.save(TEMP, 'JPEG', quality=quality)
    Temporary = Image.open(TEMP)

    # Find Difference between original and temporary
    Difference = ImageChops.difference(OriginalImage, Temporary)

    # Find the max value of color band in image
    Extrema = Difference.getextrema()
    MaximumDifference = Extrema[1]
    # print('Extrema: %d' % (MaximumDifference), end=' ')
    Scale = 255.0 / MaximumDifference

    # Enhance the image based on that scale
    Difference = ImageEnhance.Brightness(Difference).enhance(Scale)

    # Fetch the histogram of the difference image (Count of color pixels)
    Lists = Difference.histogram(mask=None, extrema=None)

    # Calculate Threshold by keeping last 75 pixels
    Pixels = 0
    Width, Height = OriginalImage.size
    for i in range(255, 1, -1):
        if Pixels + Lists[i] <= (Height * Width / 1200):
            Pixels += Lists[i]
        else:
            Threshold = i + 1
            # print('Threshold: %d' % (Threshold), end=' ')
            break

    # Apply Threshold
    BlackWhite = Difference.point(lambda x: 0 if x < Threshold else 255, '1')
    BlackWhite.save(TEMP, 'JPEG', quality=quality)

    # Load the binary image
    image = cv2.imread(TEMP, cv2.IMREAD_GRAYSCALE)
    
    kernel = np.ones((4, 4), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations = 1)
    
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(dilated_image, kernel, iterations = 1)
    
    # Apply morphological opening
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(eroded_image, cv2.MORPH_OPEN, kernel, iterations=2)

    blurred = cv2.GaussianBlur(opening, (3, 3), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 10, 50)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of pixels in each detected edge (contour)
    edge_pixel_counts = [cv2.contourArea(contour) for contour in contours]

    ln = len(edge_pixel_counts)
    s = sum(edge_pixel_counts) + max(edge_pixel_counts) ** 2
        
    return s/ln
