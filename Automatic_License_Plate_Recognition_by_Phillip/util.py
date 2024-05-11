"""
Project: Automatic License Plate Recognition (ALPR)
Aim: Initialize and define the functions which will be used to detect vehicles, to detect and recognize license plates, and to store the detection information of the detected vehicles and license plates on a csv file.
"""

import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'vehicle_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for vehicle_id in results[frame_nmr].keys():
                print(results[frame_nmr][vehicle_id])
                if 'car' in results[frame_nmr][vehicle_id].keys() and \
                   'license_plate' in results[frame_nmr][vehicle_id].keys() and \
                   'text' in results[frame_nmr][vehicle_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            vehicle_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][vehicle_id]['car']['bbox'][0],
                                                                results[frame_nmr][vehicle_id]['car']['bbox'][1],
                                                                results[frame_nmr][vehicle_id]['car']['bbox'][2],
                                                                results[frame_nmr][vehicle_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][vehicle_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][vehicle_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][vehicle_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][vehicle_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][vehicle_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][vehicle_id]['license_plate']['text'],
                                                            results[frame_nmr][vehicle_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # First, check the total number of characters on the license plate
    if len(text) != 7: # check if the license plate format does not have 7 character. If yes, then return False
        return False

    # If the condition of the total number of characters on the license plate is satisfied, then check the types of characters (letter/number) of each character on the license plate
    # if the FIRST character is a letter, SECOND character is a letter, THIRD character is a number, FOURTH character is a number, 
    # FIFTH character is a letter, SIXTH character is a letter, and SEVENTH character is a letter, then return True. 
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False # if either one condition is not satisfied, then return False.


def format_license(text):
    """
    Going through all the characters on the license plate, one at a time. Format the license plate text by converting each character using the mapping dictionaries (from letter to number, or vice versa, based on the letter format of the license plate).

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop) # input the binary cropped license plate (image) to the OCR to detect text (a set of characters with the combination of letters and numbers)

    for detection in detections: # many detections/texts are detected in this image
        bbox, text, score = detection # For each detection/text, save 3 values (bounding box of the text, text itself, and the confidence score of the text) of its detection information into 3 different variables respectively. 

        text = text.upper().replace(' ', '') # ".upper()" convert the text to uppercase, "replace(' ', '')" remove all the white spaces (blanks). Then the processed text is stored in the variable called "text" again

        if license_complies_format(text): # if the license plate text complies with the required format
            return format_license(text), score # return the formatted/improved text (to ensure each character of the text is shown with correct type) of the license plate and its confidence score

    return 0, 0 # if the license plate text DOES NOT comply with the required format, return 0 as the license plate text


def get_car(license_plate, vehicle_track_ids): # concept: find the bounding box of the vehicle which encloses a bounding box of license plate
    # "license_plate" contains the detection information of a single detected license plate which being processed in the for loop in "main.py" at the moment; "vehicle_track_ids" contains the detection information of all the detected vehicles on a frame
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)): # iterate the detection information (bounding box coordinates & vehicle ID of the detected vehicle) of all the detected vehicles on a frame 
        xcar1, ycar1, xcar2, ycar2, vehicle_id = vehicle_track_ids[j] # save all elements (5 values) of "vehicle_track_ids" into 5 different variables respectively. "vehicle_track_ids" contains the detection information of all the detected vehicles on a frame, such that "vehicle_track_ids[0]" contains the detection information of 1st detected vehicles, "vehicle_track_ids[1]" contains the detection information of 2nd detected vehicles, and so on. 

        # the coordinate system of the image: x-axis: -> means positive; <- means negative; y-axis: Towards upwards means negative, towards downwards means positive; the origin is located at the upper left corner of the image.
        # x1 is the upper left corner coordinate of the license plate bounding box, x2 is the bottom right corner coordinate of the license plate bounding box, xcar1 is the upper left corner coordinate of the vehicle bounding box, xcar2 is the bottom right corner coordinate of the vehicle bounding box
        # "x1 > xcar1" means, by only considering horizontal/x-axis, if the upper left corner coordinate of the license plate bounding box is greater than the one of the vehicle bounding box (the upper left corner coordinate of the license plate bounding box is located at the right of the one of the vehicle bounding box)
        # "y1 > ycar1" means, by only considering vertical/y-axis, if the upper left corner coordinate of the license plate bounding box is greater than the one of the vehicle bounding box (the upper left corner coordinate of the license plate bounding box is located below of the one of the vehicle bounding box)
        # "x2 < xcar2" means, by only considering horizontal/x-axis, if the bottom right corner coordinate of the license plate bounding box is smaller than the one of the vehicle bounding box (the bottom right corner coordinate of the license plate bounding box is located at the left of the one of the vehicle bounding box)
        # "y2 < ycar2" means, by only considering vertical/y-axis, if the bottom right corner coordinate of the license plate bounding box is smaller than the one of the vehicle bounding box (the bottom right corner coordinate of the license plate bounding box is located above of the one of the vehicle bounding box)

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2: # if all of these conditions are satisfied, it means this license plate bounding box is enclosed by this vehicle bounding box
            vehicle_indx = j # record the index of the detected vehicle in "vehicle_track_ids" whose bounding box encloses the license plate bounding box
            foundIt = True # serves as a flag to indicate that the license plate is successfully assigned to a specific vehicle
            break # after record the index & set the flag to 1, exit this for loop.

    if foundIt: # for the case the license plate is successfully assigned to a specific vehicle
        return vehicle_track_ids[vehicle_indx] # return the detection information of the (car_indx)th detected vehicles stored in "vehicle_track_ids"

    return -1, -1, -1, -1, -1 # for the case the license plate is NOT successfully assigned to a specific vehicle, return 5 of -1
