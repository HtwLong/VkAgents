import os
from typing import Dict, List
import urllib.request
from pprint import pprint
import requests
from vision_utils import semkg_api
import os
from typing import List, Dict

def query(query_string, token=""):
    """
    Executes a SPARQL query against the VisionKG endpoint.
    """
    try:
        response = requests.get(
            'https://vision.semkg.org/sparql',
            params={"query": query_string, "token": token},
        )
        response.raise_for_status()
        _data = response.json()
        
        data = []
        # print("Query Result:")
        # pprint.pprint(_data) # Optional: verify data visually
        
        if 'results' in _data and 'bindings' in _data['results']:
            for result in _data['results']['bindings']:
                tmp = {}
                for key in result.keys():
                    tmp[key] = result[key]['value']
                data.append(tmp)
        else:
            print("VisionKG returned an unexpected response:")
            pprint(_data)
        return data
    except Exception as e:
        print(f"Query failed: {e}")
        return []

def prepare_data(images, DATA_ROOT_PATH=None):
    if not DATA_ROOT_PATH:
        print("DATA path did not set! Path will set default at /tmp")
        DATA_ROOT_PATH = "/tmp"
    dataset_list = {}
    for image in images:
        tmp = image['image_path'].split("/")
        filename = tmp.pop()
        # --- FIX 1: Use os.path.join for safe path creation ---
        # Old: path = DATA_ROOT_PATH + "/".join(tmp)
        directory_path = os.path.join(DATA_ROOT_PATH, *tmp)

        # Define the full path for the file
        full_file_path = os.path.join(directory_path, filename)
        
        dataset = tmp[-1]
        if dataset not in dataset_list:
            dataset_list[dataset] = {'path': directory_path,
                                     'missing': []}

        if not os.path.isdir(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        if not os.path.exists(full_file_path):
            isSuccess = False
            if dataset == "visual_genome":
                # perform download
                pass
            else:
                if image.get('url'):
                    print("Image", image['file_name'],
                          'is available online. Downloading..')
                    try:
                        urllib.request.urlretrieve(image['url'], full_file_path)
                        isSuccess = True
                        print(image['file_name'], "downloaded!")
                    except:
                        isSuccess = False
                        print(image['file_name'], "download failed!")
            if not isSuccess:
                dataset_list[tmp[-1]]['missing'].append(image['file_name'])
    for dataset in dataset_list:
        if len(dataset_list[dataset]['missing']) > 0:
            print("\nThe following images of the ", dataset, "dataset are not exists. Please download and put them at",
                  dataset_list[dataset]['path'], ":")
            print(", ".join(dataset_list[dataset]['missing']))
            print("")



def get_multi_class_stats(classes: list) -> dict:
    """
    Retrieves dataset statistics for multiple classes using a single 
    SPARQL query optimized with the VALUES clause.
    
    Args:
        classes (list): A list of strings, e.g., ["cat", "dog", "bird"].

    Returns:
        dict: A nested dictionary where keys are class names and values are 
              dictionaries of dataset counts.
              Example: {'cat': {'coco': 300}, 'dog': {'voc': 100}}
    """
    
    if not classes:
        return {}

    # Initialize the output structure with empty dicts for all requested classes
    all_stats = {cls: {} for cls in classes}

    # Format the classes into a SPARQL VALUES string: "cat" "dog" "bird"
    values_string = " ".join([f'"{cls}"' for cls in classes])

    # 1. Construct the Main Query
    query_string = f"""
    PREFIX cv: <http://vision.semkg.org/onto/v0.1/>
    PREFIX schema: <http://schema.org/>

    SELECT ?targetLabel ?datasetName (COUNT(DISTINCT ?image) AS ?count)
    WHERE {{
        # Inject the list of classes directly into the query engine's execution plan
        VALUES ?targetLabel {{ {values_string} }} 
        
        ?image cv:hasAnnotation ?annotation .
        ?annotation a cv:ObjectDetectionAnnotation .
        ?annotation cv:hasLabel ?lbl .
        ?lbl cv:label ?targetLabel .
        ?image schema:isPartOf / schema:name ?datasetName .
    }}
    GROUP BY ?targetLabel ?datasetName
    ORDER BY ?targetLabel DESC(?count)
    """
    
    print(f"Querying VisionKG for {len(classes)} classes using VALUES...")

    # 2. Execute the Query
    raw_result = query(query_string)

    # 3. Parse Results into Nested Dictionary
    bindings = []
    if isinstance(raw_result, dict) and 'results' in raw_result:
        bindings = raw_result['results']['bindings']
    elif isinstance(raw_result, list):
        bindings = raw_result

    for row in bindings:
        # Helper to extract 'value' if it's a dict, or use the item directly
        def get_val(item):
            return item.get('value') if isinstance(item, dict) else item

        label = get_val(row.get('targetLabel'))
        d_name = get_val(row.get('datasetName'))
        count_val = get_val(row.get('count'))

        if label and d_name and count_val:
            try:
                # Map the results back to the initialized dictionary
                if label in all_stats:
                    all_stats[label][d_name] = int(count_val)
            except ValueError:
                pass

    return all_stats


def get_datasets():
    return [
        "ACDC_det_val_night",
        "CUB-200-2011_cls_test",
        "CUB-200-2011_cls_train",
        "LVIS_det_train",
        "LVIS_det_val",
        "SOP_cls_test",
        "SOP_cls_train",
        "UA-DETRAC_det",
        "bdd_100k_det_train",
        "bdd_100k_det_val",
        "caltech101_cls",
        "cars196_cls_test",
        "cars196_cls_train",
        "cars196_det_test",
        "cifar100_cls_test",
        "cifar10_cls_test",
        "cifar10_cls_train",
        "cityscapes_det_val",
        "cityscapes_inseg_val",
        "coco2017_det_val",
        "imageNet-1K_cls_train",
        "imageNet-1K_cls_val",
        "mapillary_v1.2_det_train",
        "mapillary_v1.2_det_val",
        "mnist_cls_test",
        "mnist_cls_train",
        "objects365_det_val",
        "openimages_challenge_2019_det_train",
        "voc0712_det_val",
        "voc07_det_test",
        "voc07_det_val",
        "voc12_det_train",
        "voc12_det_val",
        "voc12_inseg_train"]


def visionkg2cocoDet(query_bindings: List[Dict], 
                     global_image_map: Dict = None, 
                     global_category_map: Dict = None,
                     global_anno_id_counter: List[int] = None) -> Dict:
    """
    Converts VisionKG SPARQL bindings to COCO format using global registries
    to maintain consistent IDs across multiple batches.
    """
    
    # Initialize globals if not provided (safe default for single-run use)
    if global_image_map is None: global_image_map = {}
    if global_category_map is None: global_category_map = {}
    # We use a list for the counter so it can be mutable (passed by reference)
    if global_anno_id_counter is None: global_anno_id_counter = [0]

    coco_annotations = []
    coco_images_info = []
    
    # We track images added *in this specific batch* to avoid adding the same image info 
    # twice to the output list, even if it already exists in the global map.
    batch_processed_images = set()

    for anno in query_bindings:
        
        image_name = anno['imageName']
        dataset_name = anno['datasetName']
        label_name = anno['labelName']
        
        # --- 1. Handle Image ID ---
        if image_name not in global_image_map:
            # Create new ID
            global_image_map[image_name] = len(global_image_map) + 1
            
            # Create Image Info
            image_height = int(anno['imageHeight'])
            image_width = int(anno['imageWidth'])
            image_url = f"https://vision-api.semkg.org/api/image?image=/{dataset_name}/{image_name}"
            
            image_info = {
                'id': global_image_map[image_name],
                'file_name': image_name,
                'dataset': dataset_name,
                'height': image_height,
                'width': image_width,
                'url': image_url,
                'image_path': os.path.join(dataset_name, image_name),
            }
            coco_images_info.append(image_info)
            batch_processed_images.add(image_name)
        
        # If image exists in global map but hasn't been added to this batch's output list yet
        # (OPTIONAL: Depending on if you want the output to contain ALL images or just NEW ones.
        # usually for merging, you only want to append new image dicts).
        
        # --- 2. Handle Category ID ---
        if label_name not in global_category_map:
            global_category_map[label_name] = len(global_category_map) + 1

        # --- 3. Handle Bounding Box ---
        box_center_x = float(anno['bbCentreX'])
        box_center_y = float(anno['bbCentreY'])
        box_height = float(anno['bbHeight'])
        box_width = float(anno['bbWidth'])
        
        image_w = int(anno['imageWidth'])
        image_h = int(anno['imageHeight'])

        # Safety check for bounds
        # (Note: Using strict assertion might crash your loop if data is bad. 
        #  Better to clamp or skip.)
        if (box_center_x + box_width > image_w) or (box_center_y + box_height > image_h):
             # You might want to print a warning here instead of crashing
             pass 

        # Increment global annotation counter
        global_anno_id_counter[0] += 1
        
        coco_annotation = {
            'id': global_anno_id_counter[0],
            'image_id': global_image_map[image_name],
            'bbox': [round(box_center_x, 2), round(box_center_y, 2), 
                     round(box_width, 2), round(box_height, 2)],
            'category_id': global_category_map[label_name],
            'iscrowd': 0,
            'area': round(box_height * box_width),
        }
        coco_annotations.append(coco_annotation)

    # Convert global categories map to COCO list format
    # We return the FULL category list every time so the latest update has everything
    coco_categories = [{'id': v, 'name': k, 'supercategory': None} 
                       for k, v in global_category_map.items()]

    return {
        'images': coco_images_info,
        'annotations': coco_annotations,
        'categories': coco_categories
    }

def visionkg_parse_classification(query_bindings: List[Dict], global_image_set: set = None) -> Dict:
    """
    Parses VisionKG SPARQL bindings into flat rows for a classification CSV,
    and returns a list of required images to download.
    """
    if global_image_set is None:
        global_image_set = set()

    images_to_download = []
    csv_rows = []

    for anno in query_bindings:
        image_name = anno['imageName']
        dataset_name = anno['datasetName']
        label_name = anno['labelName']

        # Create the relative path expected by your CocoImageDataset loader
        rel_image_path = os.path.join(dataset_name, image_name)

        # Track unique images so we don't download duplicates across batches
        if rel_image_path not in global_image_set:
            global_image_set.add(rel_image_path)
            
            image_url = f"https://vision-api.semkg.org/api/image?image=/{dataset_name}/{image_name}"
            images_to_download.append({
                'file_name': image_name,
                'url': image_url,
                'image_path': rel_image_path,
            })

        # Append flat row for the CSV
        csv_rows.append({
            'image_filename': rel_image_path,
            'labels': label_name
        })

    return {
        'images_to_download': images_to_download,
        'csv_rows': csv_rows
    }

import urllib.request
import os

def prepare_data_flat(images: list, DATA_ROOT_PATH: str = None):
    """
    Downloads images directly into the root directory without creating subfolders.
    The filename is created by flattening the relative image path (replacing '/' with '_').
    """
    if not DATA_ROOT_PATH:
        print("DATA path did not set! Path will set default at /tmp")
        DATA_ROOT_PATH = "/tmp"
        
    if not os.path.isdir(DATA_ROOT_PATH):
        os.makedirs(DATA_ROOT_PATH, exist_ok=True)

    missing_images = []
    
    for image in images:
        # Flatten the path (e.g., "coco2017_det_train/000000102096.jpg" -> "coco2017_det_train_000000102096.jpg")
        # We replace both standard slashes and backslashes just to be safe across OS environments
        flat_filename = image['image_path'].replace('/', '_').replace('\\', '_')
        full_file_path = os.path.join(DATA_ROOT_PATH, flat_filename)
        
        if not os.path.exists(full_file_path):
            is_success = False
            if image.get('url'):
                print(f"Downloading {flat_filename}...")
                try:
                    urllib.request.urlretrieve(image['url'], full_file_path)
                    is_success = True
                except Exception as e:
                    print(f"Download failed for {flat_filename}: {e}")
            
            if not is_success:
                missing_images.append(flat_filename)
                
    if missing_images:
        print(f"\nThe following images could not be downloaded to {DATA_ROOT_PATH}:")
        print(", ".join(missing_images))
    else:
        print("\nAll downloads finished successfully!")
