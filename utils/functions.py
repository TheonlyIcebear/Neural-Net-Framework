import numpy as np
# from numba import *
# from typing import *
# from numba.experimental import jitclass
from functools import partial
from PIL import Image, ImageTk, ImageDraw

class Processing:
    def iou(box, boxes):
        box_top_left = (box[:2] - (box[2:] / 2))
        box_bottom_right = (box[:2] + (box[2:] / 2))

        boxes_top_left = (boxes[:, :2] - (boxes[:, 2:] / 2))
        boxes_bottom_right = (boxes[:, :2] + (boxes[:, 2:] / 2))

        top_left_x = np.maximum(box_top_left[0], boxes_top_left[:, 0])
        top_left_y = np.maximum(box_top_left[1], boxes_top_left[:, 1])

        bottom_right_x = np.minimum(box_bottom_right[0], boxes_bottom_right[:, 0])
        bottom_right_y = np.minimum(box_bottom_right[1], boxes_bottom_right[:, 1])

        intersection = np.maximum(0, bottom_right_x - top_left_x) * np.maximum(0, bottom_right_y - top_left_y)
        box_area = (box_bottom_right[0] - box_top_left[0]) * (box_bottom_right[1] - box_top_left[1])
        boxes_area = (boxes_bottom_right[:, 0] - boxes_top_left[:, 0]) * (boxes_bottom_right[:, 1] - boxes_top_left[:, 1])

        union = box_area + boxes_area - intersection
        return intersection / (union + 1e-10)

    def draw_boxes(image, points, color):
        points_count = int(points.shape[0] / 4)


        predicted_points = np.array(points.reshape((points_count, 2, 2)))

        draw = ImageDraw.Draw(image)

        dimensions = np.array(image.size).astype(int)

        for center, distances in predicted_points:
            top_left = (center - (distances / 2)) * dimensions
            bottom_right = (center + (distances / 2)) * dimensions

            draw.line([(top_left[0], top_left[1]), (bottom_right[0], top_left[1])], fill=color, width=2)
            draw.line([(top_left[0], bottom_right[1]), (bottom_right[0], bottom_right[1])], fill=color, width=2)

            draw.line([(top_left[0], top_left[1]), (top_left[0], bottom_right[1])], fill=color, width=2)
            draw.line([(bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1])], fill=color, width=2)

class Activations:
    @staticmethod
    # @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def sigmoid(x, deriv=False):
        if deriv:
            return x * (1 - x)

        return ( 1 / ( 1 + np.exp(-x) ) )

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def tanh(x, deriv=False):
        if deriv:
            return (1 - x ** 2)

        return np.tanh(x)

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def relu(x, deriv=False):
        
        if deriv:
            return 1 * (x > 0)

        return x * (x > 0)

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def selu(x, deriv=False):

        alpha = 1.67326
        scale = 1.0507

        if deriv:
            return np.where(x <= 0, scale * alpha * np.exp(x), scale)

        return np.where(x <= 0, scale * alpha * (np.exp(x) - 1), scale * x)

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def lrelu(x, deriv=False):
        negative_slope = 10 ** -1

        if deriv:
            return 1 * (x > 0) + (negative_slope * (x < 0))

        return 1 * x * (x > 0) + (negative_slope * x * (x < 0))

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def crelu(x, deriv=False):
        if deriv:
            return (1 * ((x > 0) & (x < 1)))

        return x * ((x > 0) & (x < 1)) + (1 * (x >= 1))

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def softmax(x, deriv=False):
        if deriv:
            softmax_output = np.exp(x) / np.sum(np.exp(x))
            return softmax_output * (1 - softmax_output)

        e_x = np.exp(x)

        return e_x / e_x.sum()


class Loss:
    @staticmethod
    # @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def cross_entropy(outputs, expected_outputs, deriv=False):
        if deriv:
            return (outputs - expected_outputs)
            
        epsilon = 1e-12

        outputs = np.clip(outputs, epsilon, 1.0 - epsilon)

        return np.mean(-(expected_outputs * np.log(outputs + epsilon)))

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def mse(outputs, expected_outputs, deriv=False):
        if deriv:
            return 2 * (outputs - expected_outputs)

        if outputs.shape[0] == 0:
            return 0
        
        return np.mean((outputs - expected_outputs) ** 2)

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def smooth_l1_loss(outputs, expected_outputs, deriv=False):
        if outputs.size == 0:
            return 0

        delta = 1.0

        if deriv:
            diff = outputs - expected_outputs
            mask = np.abs(diff) <= delta
            return np.where(mask, diff, np.sign(diff) * delta)
            
        diff = outputs - expected_outputs
        return np.mean(np.where(np.abs(diff) <= delta, 0.5 * diff ** 2, delta * np.abs(diff) - 0.5 * delta))

    @staticmethod
    # @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def yolo_loss(outputs, expected_outputs, deriv=False):
        coordinate_weight = 5
        no_object_weight = 0.5
        object_weight = 1

        presence_scores = expected_outputs[::5]

        mask = np.arange(0, outputs.shape[0])
        mask = (mask % 5 != 0) & (expected_outputs[(mask // 5) * 5] == 1)

        expected_presence_scores = expected_outputs[::5]
        inactive_boxes = expected_presence_scores == 0
        active_boxes = expected_presence_scores == 1

        if deriv:
            deriv_values = np.zeros(outputs.shape)

            deriv_values[::5][inactive_boxes] = Loss.mse(outputs[::5][inactive_boxes], expected_outputs[::5][inactive_boxes], deriv=True) * no_object_weight
            deriv_values[::5][active_boxes] = Loss.mse(outputs[::5][active_boxes], expected_outputs[::5][active_boxes], deriv=True) * object_weight
            deriv_values[mask] = Loss.mse(outputs[mask], expected_outputs[mask], deriv=True) * coordinate_weight

            return deriv_values

        no_object_loss = Loss.mse(outputs[::5][inactive_boxes], expected_outputs[::5][inactive_boxes]) * no_object_weight
        object_loss = Loss.mse(outputs[::5][active_boxes], expected_outputs[::5][active_boxes]) * object_weight
        coordinate_loss = Loss.mse(outputs[mask], expected_outputs[mask]) * coordinate_weight

        loss = object_loss + no_object_loss + coordinate_loss

        return loss