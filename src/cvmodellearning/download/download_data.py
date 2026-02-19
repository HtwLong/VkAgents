import json
import csv
import os
from cvmodellearning.download.visionkg_utils import prepare_data, visionkg2cocoDet, query, visionkg_parse_classification, prepare_data_flat
from cvmodellearning.paths import data_dir, json_labels_path, csv_labels_path

def download_visionkg_mixed_datasets_detection(job_id: str, requests: list):
    """
    Sequentially downloads images and aggregates labels using a single integrated query,
    while maintaining unique IDs across all batches using global registries.
    """
    
    # --- 1. Validate Input Structure ---
    if not isinstance(requests, list):
        raise TypeError(f"Input 'requests' must be a list. Got {type(requests)}.")

    if not requests:
        print("Warning: 'requests' list is empty. Nothing to download.")
        return

    # --- 2. Initialize Global Registries (The "Memory") ---
    # These dictionaries must exist outside the loop so they persist across all batches.
    global_image_map = {}      # Maps filename -> Unique Integer ID
    global_category_map = {}   # Maps class name -> Unique Integer ID
    global_anno_id_counter = [0] # Mutable list to track Annotation IDs
    
    master_coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # --- 3. Iterate through Request List ---
    for entry in requests:
        if not isinstance(entry, dict):
            continue

        class_name = entry.get("class_name")
        sources = entry.get("sources")

        if not class_name or not sources or not isinstance(sources, list):
            continue

        # --- 4. Iterate through Sources ---
        for source in sources:
            if not isinstance(source, dict):
                continue

            dataset_name = source.get("dataset_name")
            limit = source.get("image_count")

            if not dataset_name or not isinstance(limit, int):
                continue

            print(f"\n--- Processing: Class '{class_name}' from Dataset '{dataset_name}' (Limit: {limit}) ---")

            # --- 5. Build Single Integrated Query ---
            query_string = f"""
            PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
            PREFIX schema:<http://schema.org/>
            
            SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image 
                   ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
            WHERE {{
                # --- INNER SUBQUERY (Limits by Image Count) ---
                {{
                    SELECT DISTINCT ?image ?datasetName
                    WHERE {{
                        ?image schema:isPartOf / schema:name ?datasetName .
                        FILTER regex(?datasetName, "{dataset_name}", "i")
                        
                        ?image cv:hasAnnotation ?ann .
                        ?ann cv:hasLabel/cv:label ?labelName .
                        FILTER regex(?labelName, "{class_name}", "i")
                    }}
                    LIMIT {limit}
                }}

                # --- OUTER DATA FETCHING ---
                OPTIONAL {{ ?image schema:name ?imageName }} .
                OPTIONAL {{ 
                    ?image cv:imgWidth ?imageWidth .
                    ?image cv:imgHeight ?imageHeight .
                }}

                ?image cv:hasAnnotation ?annotation .
                ?annotation cv:hasLabel/cv:label ?labelName .
                
                # Filter again to ensure we get the specific class boxes
                FILTER regex(?labelName, "{class_name}", "i")
                
                OPTIONAL {{
                    ?annotation cv:hasBox ?bbox .
                    OPTIONAL {{?bbox cv:boxHeight ?bbHeight }} .
                    OPTIONAL {{?bbox cv:boxWidth ?bbWidth }} .
                    OPTIONAL {{?bbox cv:centerX ?bbCentreX . ?bbox cv:centerY ?bbCentreY }} .
                }}
            }}
            """

            # --- 6. Execute Query ---
            print("  Querying VisionKG...")
            raw_result = query(query_string)
            
            if not raw_result:
                print(f"  No results found for {class_name} in {dataset_name}.")
                continue

            # --- 7. Convert using Global Registries ---
            # IMPORTANT: Pass the shared dictionaries here!
            partial_coco_data = visionkg2cocoDet(
                raw_result, 
                global_image_map=global_image_map, 
                global_category_map=global_category_map, 
                global_anno_id_counter=global_anno_id_counter
            )
            
            # --- 8. Download Images ---
            # 'partial_coco_data' only contains images that were NEW to the global map,
            # so we don't waste time checking files we already processed in previous loops.
            if partial_coco_data['images']:
                print(f"  Downloading {len(partial_coco_data['images'])} new images...")
                prepare_data(partial_coco_data['images'], DATA_ROOT_PATH=str(data_dir(job_id)))
            else:
                print("  No new images to download (all already exist in batch).")

            # --- 9. Aggregate Data ---
            # Extend images (safe because we only get new ones back)
            master_coco_data['images'].extend(partial_coco_data.get('images', []))
            
            # Extend annotations (safe because global counter ensures unique IDs)
            master_coco_data['annotations'].extend(partial_coco_data.get('annotations', []))
            
            # Overwrite categories (safe because 'visionkg2cocoDet' returns the FULL updated list every time)
            master_coco_data['categories'] = partial_coco_data.get('categories', [])

    # --- 10. Save Final JSON ---
    if master_coco_data['images']:
        json_path = json_labels_path(job_id)
        with open(json_path, 'w') as f:
            json.dump(master_coco_data, f, indent=4)
        print(f"\nSaved merged annotations to {json_path}")
        print(f"Total Images: {len(master_coco_data['images'])}")
        print(f"Total Annotations: {len(master_coco_data['annotations'])}")
        print(f"Categories: {[c['name'] for c in master_coco_data['categories']]}")
    else:
        print("\nNo data collected.")




def download_visionkg_mixed_datasets_classification(job_id: str, requests: list):
    """
    Sequentially downloads images and aggregates labels into a flat CSV file 
    for image classification pipelines.
    """
    if not isinstance(requests, list):
        raise TypeError(f"Input 'requests' must be a list. Got {type(requests)}.")

    if not requests:
        print("Warning: 'requests' list is empty. Nothing to download.")
        return

    # A set to keep track of images we have already processed/downloaded
    global_image_set = set()
    master_csv_rows = []

    for entry in requests:
        if not isinstance(entry, dict):
            continue

        class_name = entry.get("class_name")
        sources = entry.get("sources")

        if not class_name or not sources or not isinstance(sources, list):
            continue

        for source in sources:
            if not isinstance(source, dict):
                continue

            dataset_name = source.get("dataset_name")
            limit = source.get("image_count")

            if not dataset_name or not isinstance(limit, int):
                continue

            print(f"\n--- Processing: Class '{class_name}' from Dataset '{dataset_name}' (Limit: {limit}) ---")

            query_string = f"""
            PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
            PREFIX schema:<http://schema.org/>
            
            SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image 
                   ?labelName
            WHERE {{
                {{
                    SELECT DISTINCT ?image ?datasetName
                    WHERE {{
                        ?image schema:isPartOf / schema:name ?datasetName .
                        FILTER regex(?datasetName, "{dataset_name}", "i")
                        
                        ?image cv:hasAnnotation ?ann .
                        ?ann cv:hasLabel/cv:label ?labelName .
                        FILTER regex(?labelName, "{class_name}", "i")
                    }}
                    LIMIT {limit}
                }}

                OPTIONAL {{ ?image schema:name ?imageName }} .
                OPTIONAL {{ 
                    ?image cv:imgWidth ?imageWidth .
                    ?image cv:imgHeight ?imageHeight .
                }}

                ?image cv:hasAnnotation ?annotation .
                ?annotation cv:hasLabel/cv:label ?labelName .                
                FILTER regex(?labelName, "{class_name}", "i")
            }}
            """

            print("  Querying VisionKG...")
            raw_result = query(query_string)
            
            if not raw_result:
                print(f"  No results found for {class_name} in {dataset_name}.")
                continue

            # Parse results directly into flat CSV rows and a download list
            parsed_data = visionkg_parse_classification(
                raw_result, 
                global_image_set=global_image_set
            )
            
            if parsed_data['images_to_download']:
                print(f"  Downloading {len(parsed_data['images_to_download'])} new images...")
                prepare_data(parsed_data['images_to_download'], DATA_ROOT_PATH=str(data_dir(job_id)))
            else:
                print("  No new images to download (all already exist in batch).")

            master_csv_rows.extend(parsed_data['csv_rows'])

    # Save to CSV using the exact headers your pipeline expects
    if master_csv_rows:
        csv_path = csv_labels_path(job_id)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            # Matches headers exactly for `prepare_data_step`
            writer = csv.DictWriter(f, fieldnames=['image_filename', 'labels'])
            writer.writeheader()
            writer.writerows(master_csv_rows)
            
        print(f"\nSaved merged annotations to {csv_path}")
        print(f"Total Rows in CSV: {len(master_csv_rows)}")
    else:
        print("\nNo data collected.")


def download_visionkg_images_flat(job_id: str, requests: list):
    """
    Sequentially queries VisionKG for images and downloads them into a single 
    directory with flattened filenames. No annotations are processed.
    """
    if not isinstance(requests, list):
        raise TypeError(f"Input 'requests' must be a list. Got {type(requests)}.")

    if not requests:
        print("Warning: 'requests' list is empty. Nothing to download.")
        return

    # Track unique images to prevent duplicate downloads
    global_image_set = set()
    images_to_download = []

    for entry in requests:
        if not isinstance(entry, dict):
            continue

        class_name = entry.get("class_name")
        sources = entry.get("sources")

        if not class_name or not sources or not isinstance(sources, list):
            continue

        for source in sources:
            if not isinstance(source, dict):
                continue

            dataset_name = source.get("dataset_name")
            limit = source.get("image_count")

            if not dataset_name or not isinstance(limit, int):
                continue

            print(f"\n--- Fetching Images: Class '{class_name}' from Dataset '{dataset_name}' (Limit: {limit}) ---")

            # Highly simplified query: We only need the image and dataset names
            query_string = f"""
            PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
            PREFIX schema:<http://schema.org/>
            
            SELECT DISTINCT ?datasetName ?imageName
            WHERE {{
                ?image schema:isPartOf / schema:name ?datasetName .
                FILTER regex(?datasetName, "{dataset_name}", "i")
                
                ?image cv:hasAnnotation ?ann .
                ?ann cv:hasLabel/cv:label ?labelName .
                FILTER regex(?labelName, "{class_name}", "i")
                
                OPTIONAL {{ ?image schema:name ?imageName }} .
            }}
            LIMIT {limit}
            """

            print("  Querying VisionKG...")
            raw_result = query(query_string)
            
            if not raw_result:
                print(f"  No results found for {class_name} in {dataset_name}.")
                continue

            for row in raw_result:
                dataset_n = row.get('datasetName')
                img_name = row.get('imageName')
                
                if not dataset_n or not img_name:
                    continue
                    
                # Creating a clean relative path (standardizing on forward slash for the URL)
                rel_image_path = f"{dataset_n}/{img_name}"
                
                if rel_image_path not in global_image_set:
                    global_image_set.add(rel_image_path)
                    
                    image_url = f"https://vision-api.semkg.org/api/image?image=/{dataset_n}/{img_name}"
                    images_to_download.append({
                        'url': image_url,
                        'image_path': rel_image_path,
                    })

    if images_to_download:
        print(f"\nStarting download for {len(images_to_download)} unique images...")
        prepare_data_flat(images_to_download, DATA_ROOT_PATH=str(data_dir(job_id)))
    else:
        print("\nNo new images to download.")