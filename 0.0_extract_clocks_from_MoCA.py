import numpy as np
import math
import cv2
import os
from pdf2image import convert_from_path

'''Extract the clock images from full MoCA pdfs

    (Caveat - this script is very sensitive and likely not applicable to all types of scans. It is very likely that 
    fresh data would need to be manually cropped - even in practice this method failed on around 5% of the total scans.
    Consider this as an assistive tool and not necessarily a fully automated step in the pipeline)

   base_directory = directory containing ONLY pdf files of scanned MoCAs
   save_dir = directory to save the cropped clock images as jpgs'''

# -------------------------------------------------------------------------------------------------------------------- #
# Change these strings to the desired source and destination directories
base_directory = "data/raw_data"
save_dir = "data/cropped_images"

# -------------------------------------------------------------------------------------------------------------------- #

top_left_filter = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

top_left_filter = top_left_filter / np.sum(top_left_filter) * -1

top_right_filter = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1, 1, 0, 0]])

bottom_left_filter = np.array([[0, 0, 1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]])

bottom_right_filter = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  0]])


bottom_right_filter = bottom_right_filter / np.sum(bottom_right_filter) * -1

def rotate_image(image, angle):
    image_center = (image.shape[0]/2, image.shape[1]/2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    return cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

#######################################################################################################################

def get_lines(image):
    # Threshold and get edges
    _, threshed_cv = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(threshed_cv, 50, 150)

    # Get lines
    minLineLength = 200
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=110, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    return lines

########################################################################################################################

def get_corner(image, filter, offset_x=0, offset_y=0):
    coords = [0, 0]

    # Filter the image with a corner filter and find the maximum result.
    _, threshed = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    filtered = cv2.filter2D(threshed, -1, filter)
    filter_coords = np.unravel_index(np.argmax(filtered), filtered.shape)

    # Adjust the returned coords to give the location in the full image, if necessary
    coords[0] += filter_coords[0] + offset_x
    coords[1] += filter_coords[1] + offset_y

    return coords

########################################################################################################################


if __name__ == '__main__':

    # Check if desired save directory exists
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    counter = 0
    num_errors = 0

    print("Processing files in directory: ", base_directory)
    for filename in os.listdir(base_directory):

        try:
            # Load pdf file
            path = base_directory + "/" + filename
            pages = convert_from_path(path)

            # The desired page has more ink on it than all the others
            # So the total of its pixel values will be the smallest (black = 0, white = 255)
            smallest = np.inf
            image = None
            for page in pages:
                convert = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
                white_value = np.sum(convert)
                if white_value < smallest:
                    image = page
                    smallest = white_value

            image = np.array(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)[1]

            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            biggest_area = 0
            best_contour = None

            for c in cnts:
                area = cv2.contourArea(c)
                if area > biggest_area:
                    biggest_area = area
                    best_contour = c

            #print(biggest_area)
            x, y, w, h = cv2.boundingRect(best_contour)
            cropped = image[y:y+h+1, x:x+w+1]
            #cropped = cropped[:, cropped.shape[1]/2:]
            x_split_point = int(cropped.shape[1] / 2) + 200
            y_split_point = int(3 * cropped.shape[0] / 4) - 100
            cropped = cropped[:y_split_point, x_split_point:]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)[1]

            # Get straight lines using Hough transforms
            lines = get_lines(thresh)

            # Find top and bottom lines to get angle of rotation
            bottom_line = None
            top_line = None
            highest = 0
            lowest = 700
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # Check that this is a horizontal line
                    if np.abs(y2 - y1) < np.abs(x2 - x1):
                        if y2 > highest:
                            bottom_line = line[0]
                            highest = y2
                        if y1 < lowest:
                            top_line = line[0]
                            lowest = y1

            # Get angle from bottom line
            delta_x = bottom_line[2] - bottom_line[0]
            delta_y = bottom_line[1] - bottom_line[3]
            radians = math.atan2(delta_y, delta_x)
            bottom_degrees = math.degrees(radians)

            # Get angle from top line
            delta_x = top_line[2] - top_line[0]
            delta_y = top_line[1] - top_line[3]
            radians = math.atan2(delta_y, delta_x)
            top_degrees = math.degrees(radians)

            # Get average angle across both lines
            degrees = (bottom_degrees + top_degrees) / 2.0

            # Rotate image to correct scanning errors
            rotated = rotate_image(cropped, -degrees)

            center_x = int(rotated.shape[0] / 2)
            center_y = int(rotated.shape[1] / 2)
            q1_x = int(center_x / 2)
            q1_y = int(center_y / 2)
            q3_x = int(center_x * 1.5)
            q3_y = int(center_y * 1.5)

            top_left = get_corner(rotated[:q1_x, :q1_y], top_left_filter)
            # bottom_right = get_corner(rotated[q3_x:, q3_y:], bottom_right_filter, offset_x=q3_x, offset_y=q3_y)
            # print(top_left, bottom_right)

            # Reduce image using the upper left corner point as determined through filtering.
            # Box size determined through good automated extraction in previous iterations
            reduced = rotated[top_left[0]:top_left[0] + 622, top_left[1]:top_left[1] + 444, :]
            # print(reduced.shape)

            # Reduce image further to remove edges, instructions, and scoring
            reduced = reduced[72:-50, 8:-8]
            reduced = reduced[:, 12:-15]
            _, threshed = cv2.threshold(reduced, 240, 255, cv2.THRESH_BINARY)

            # Starting from the bottom, trim each row until there is just whitespace remaining.
            i = len(reduced) - 1
            while True:
                if np.mean(threshed[i]) != 255:
                    reduced = reduced[:-1]
                    i -= 1
                    if i < 300:
                        #print("Found score/clock interference for file: ", filename)
                        raise Exception
                else:
                    break

            i = len(reduced[0]) - 1
            while True:
                if np.mean(threshed[:, i]) < 200:
                    reduced = reduced[:, :-5]
                    i -= 1

                else:
                    break

            i = 0
            while True:
                if np.mean(threshed[:, i]) < 200:
                    reduced = reduced[:, 5:]
                    i += 1

                else:
                    break



            if 0 in reduced.shape:
                #print("Bad filter for file: ", filename)
                continue

            # Resize the final product to a standard size
            dim = (100, 125)
            #reduced = cv2.resize(reduced, dim, interpolation=cv2.INTER_AREA)

            # Save the reduced images in the directory specified at the beginning
            cv2.imwrite(save_dir + "/" + filename[:-4] + ".jpg", reduced)

            # Increment counter
            counter += 1
            # Print message every n images for illusion of progress
            if counter % 50 == 0:
                print("Processed", counter, "PDFs")

        except Exception as e:
            num_errors += 1
            print("Something went wrong processing file: ", filename)
            #print(e)

    print(num_errors)
