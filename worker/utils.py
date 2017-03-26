import tempfile
import io
import codecs
import zipfile
import os

import jprops
import numpy
from sklearn.metrics import confusion_matrix


def convert_values_to_string(variables):
    """
    Converts the values in the dictionary to the string type.

    :param variables: the variables dictionary
    :rtype: dict[str, object]

    :return: the variables dictionary with string values
    :rtype: dict[str, str]
    """
    if not variables:
        return {}
    return dict((k, str(v)) for k, v in variables.items())


def create_temporary_properties_file(variables):
    """
    Creates a .properties file with the properties provided as a dictionary.

    :param variables: the variables dictionary
    :type variables: dict[str, str]

    :return: the .properties file
    :rtype: file
    """
    properties_file = tempfile.NamedTemporaryFile()
    jprops.store_properties(properties_file, variables)
    return properties_file


def create_zip_base64_string(paths):
    """
    Creates a zip file encoded as a base64 string.

    :param paths: the file paths to include.
    :type paths: list[str]

    :return: the base64 string
    :rtype: str
    """
    zip_stream = io.BytesIO()
    zip_writer = zipfile.ZipFile(zip_stream, mode='w', compression=zipfile.ZIP_DEFLATED)

    for path in paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                archive_name = os.path.basename(path)
                zip_writer.write(path, archive_name)
            elif os.path.isdir(path):
                for root, directories, files in os.walk(path):
                    archive_root = os.path.relpath(root, path)
                    for file in files:
                        file_path = os.path.join(root, file)
                        archive_name = os.path.join(archive_root, file)
                        zip_writer.write(file_path, archive_name)

    zip_writer.close()

    zip_stream_base64 = codecs.encode(zip_stream.getvalue(), 'base64')

    return zip_stream_base64.decode('utf-8')


def extract_zip_base64_string(string, path):
    """
    Extract a zip file encoded as a base64 string to directory.

    :param string: the base64 string
    :type string: str

    :param path: the destination path.
    :type path: str
    """
    # Decodes the string from base64.
    base64_string = string.encode('utf-8')
    zip_string = codecs.decode(base64_string, 'base64')

    zip_stream = io.BytesIO(zip_string)
    zip_reader = zipfile.ZipFile(zip_stream, mode='r', compression=zipfile.ZIP_DEFLATED)

    zip_reader.extractall(path)


def read_values_from_file(file):
    """
    Reads a file containing values, one per line.

    :param file: the file containing the values, one per line
    :type file: file

    :return: a list of predictions
    :rtype: object
    """
    return numpy.genfromtxt(file, dtype=None)


def compute_confusion_matrix(actual_values, predicted_values, class_values_order):
    naive_confusion_matrix = confusion_matrix(
        actual_values,
        predicted_values,
        class_values_order
    )
    true_positives = naive_confusion_matrix[1][1]
    false_positives = naive_confusion_matrix[0][1]
    true_negatives = naive_confusion_matrix[0][0]
    false_negatives = naive_confusion_matrix[1][0]

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
    }


def convert_string(value, value_type):
    """
    Converts a string to a value according to a given type.

    :param value: the value to convert
    :rtype: str

    :param value_type: the destination between 'integer', 'real' and 'string'
    :type value_type: str

    :return: the value converted
    :rtype: object
    """
    if value_type == 'integer':
        return int(value)
    elif value_type == 'real':
        return float(value)
    elif value_type == 'text':
        return value
