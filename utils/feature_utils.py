import numpy as np

''' A set of utility functions used for computing various aspects of visual clock features
'''

def smooth_contour(points_list):
    for i, point1 in enumerate(points_list):
        # Restart if unstable, point has been deleted

        # Get adjacent points
        if i - 1 < 0:
            left = -1
        else:
            left = i - 1

        if i + 1 >= len(points_list):
            right = 0
        else:
            right = i + 1

        left_dist = np.linalg.norm(points_list[i] - points_list[left])
        right_dist = np.linalg.norm(points_list[i] - points_list[right])

        if left_dist < right_dist:
            min_dist = left_dist
            closest_point = left
        else:
            min_dist = right_dist
            closest_point = right
        for j, point2 in enumerate(points_list):
            # Restart full while loop if unstable, point has been deleted
            if i == j:
                continue
            # See if there is a closer point, if so remove previous closest point and restart
            if np.linalg.norm(points_list[i] - points_list[j]) < min_dist and j != closest_point:
                reduced_points = np.delete(points_list, [closest_point], axis=0)
                return reduced_points, False

    return points_list, True

########################################################################################################################
def determine_overlap(rect1, rect2):
    # Check if either rectangle is a line
    if (rect1[0] == rect1[2]) or (rect1[1] == rect1[3]) or (rect2[0] == rect2[2]) or (rect2[1] == rect2[3]):
        return False

    # If one rectangle is fully left of another, no intersection
    if(rect1[0] >= rect2[2] or rect2[0] >= rect1[2]):
        return False

    # If one rectangle is fully above another, no intersection
    if(rect1[1] >= rect2[3] or rect2[1] >= rect1[3]):
        return False

    return True

########################################################################################################################
def get_maximum_bounding(rect1, rect2):
    x1, x2, y1, y2 = 0, 0, 0, 0
    if rect1[0] <= rect2[0]:
        x1 = rect1[0]
    else:
        x1 = rect2[0]

    if rect1[1] <= rect2[1]:
        y1 = rect1[1]
    else:
        y1 = rect2[1]

    if rect1[2] >= rect2[2]:
        x2 = rect1[2]
    else:
        x2 = rect2[2]

    if rect1[3] >= rect2[3]:
        y2 = rect1[3]
    else:
        y2 = rect2[3]

    return [x1, y1, x2, y2]

########################################################################################################################
def area_of_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return 0
    else:
        return w * h

########################################################################################################################