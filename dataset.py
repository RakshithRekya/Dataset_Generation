import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    info = {}
    for child in root:
        if child.tag == 'object':
            for attribute in child:
                info[attribute.tag] = attribute.text
    return info

def create_dataset(base_path):
    columns = ['image_name', 'name', 'gender', 'black_hair', 'mustache', 'glasses', 'beard']
    data = []  # List to hold all data dictionaries

    for class_folder in os.listdir(base_path):
        class_path = os.path.join(base_path, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith('.txt'):
                    try:
                        image_info = parse_xml(os.path.join(class_path, file))
                        image_info['image_name'] = file.replace('.txt', '.png')
                        data.append(image_info)
                    except ET.ParseError:
                        print(f"Could not parse {file}, invalid XML.")
                    except Exception as e:
                        print(f"An error occurred with {file}: {e}")

    dataset = pd.DataFrame(data, columns=columns)
    return dataset


base_path = os.getcwd()
dataset = create_dataset(base_path)
dataset.to_csv('image_dataset.csv', index=False)
