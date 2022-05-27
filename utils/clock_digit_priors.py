from scipy.stats import norm
import numpy as np

''' Defines a set of prior probabilities for a bounding box to contain a specific digit based on its' angle from the
    centerpoint of the clock
'''

def num_to_angle(num):
    if num == 3:
        return 0
    if num == 2:
        return 30
    if num == 1:
        return 60
    if num == 12:
        return 90
    if num == 11:
        return 120
    if num == 10:
        return 150
    if num == 9:
        return 180
    if num == 8:
        return 210
    if num == 7:
        return 240
    if num == 6:
        return 270
    if num == 5:
        return 300
    if num == 4:
        return 330


# Given an angle (measured in degrees counterclockwise from positive x-axis) give a probability
def get_angle_priors(angle, sigma):
    # priors is a list with values corresponding to the probs of digits 0-9
    priors = []

    one_prob = norm.pdf(angle, loc=num_to_angle(1), scale=sigma)
    two_prob = norm.pdf(angle, loc=num_to_angle(2), scale=sigma)
    three_prob = norm.pdf(angle, loc=num_to_angle(3), scale=sigma)
    four_prob = norm.pdf(angle, loc=num_to_angle(4), scale=sigma)
    five_prob = norm.pdf(angle, loc=num_to_angle(5), scale=sigma)
    six_prob = norm.pdf(angle, loc=num_to_angle(6), scale=sigma)
    seven_prob = norm.pdf(angle, loc=num_to_angle(7), scale=sigma)
    eight_prob = norm.pdf(angle, loc=num_to_angle(8), scale=sigma)
    nine_prob = norm.pdf(angle, loc=num_to_angle(9), scale=sigma)
    ten_prob = norm.pdf(angle, loc=num_to_angle(10), scale=sigma)
    eleven_prob = norm.pdf(angle, loc=num_to_angle(11), scale=sigma)
    twelve_prob = norm.pdf(angle, loc=num_to_angle(12), scale=sigma)

    priors.append(ten_prob)
    priors.append(max(one_prob, ten_prob, eleven_prob, twelve_prob))
    priors.append(max(two_prob, twelve_prob))
    priors.append(three_prob)
    priors.append(four_prob)
    priors.append(five_prob)
    priors.append(six_prob)
    priors.append(seven_prob)
    priors.append(eight_prob)
    priors.append(nine_prob)

    return np.array(priors)
