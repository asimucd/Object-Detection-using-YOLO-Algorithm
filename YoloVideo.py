import cv2
import numpy as np

# We are not going to bother with objects less than 50% probability
THRESHOLD = 0.5
# The lower the value: the fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.4
YOLO_IMAGE_SIZE = 320

def find_objects(model_outputs):
    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > THRESHOLD:
                w, h = int(prediction[2] * YOLO_IMAGE_SIZE), int(prediction[3] * YOLO_IMAGE_SIZE)
                # The center of the bounding box (we should transform these values)
                x, y = int(prediction[0] * YOLO_IMAGE_SIZE - w / 2), int(prediction[1] * YOLO_IMAGE_SIZE - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)

    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values

def show_detected_objects(img, bounding_box_ids, all_bounding_boxes, class_ids, confidence_values, width_ratio, height_ratio):
    if len(bounding_box_ids) > 0:
        for index in bounding_box_ids.flatten():
            bounding_box = all_bounding_boxes[index]
            x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
            # We have to transform the locations and coordinates because the resized image
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)

            # OpenCV deals with BGR (blue, green, red) (255, 0, 0) is the blue color
            # We are not going to detect every object, just PERSON and CAR
            if class_ids[index] == 2:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                class_with_confidence = 'CAR ' + str(int(confidence_values[index] * 100)) + '%'
                cv2.putText(img, class_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)

            if class_ids[index] == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                class_with_confidence = 'PERSON ' + str(int(confidence_values[index] * 100)) + '%'
                cv2.putText(img, class_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)

capture = cv2.VideoCapture('yolo_test.mp4')

# There are 80 (90) possible output classes
# 0: person - 2: car - 5: bus
classes = ['car', 'person', 'bus']

neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# Define whether we run the algorithm with CPU or with GPU
# WE ARE GOING TO USE CPU !!!
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    # After reading a frame (so basically one image) we just have to do
    # the same operations as with single images !!!
    frame_grabbed, frame = capture.read()

    if not frame_grabbed:
        break

    original_width, original_height = frame.shape[1], frame.shape[0]

    # The image into a BLOB [0-1] RGB - BGR
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
    neural_network.setInput(blob)

    layer_names = neural_network.getLayerNames()
    # YOLO network has 3 output layers - note: these indexes are starting with 1
    output_layer_indices = neural_network.getUnconnectedOutLayers().flatten()
    output_names = [layer_names[i - 1] for i in output_layer_indices]

    outputs = neural_network.forward(output_names)
    predicted_objects, bbox_locations, class_label_ids, conf_values = find_objects(outputs)
    show_detected_objects(frame, predicted_objects, bbox_locations, class_label_ids, conf_values,
                     original_width / YOLO_IMAGE_SIZE, original_height / YOLO_IMAGE_SIZE)

    cv2.imshow('YOLO Algorithm', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
