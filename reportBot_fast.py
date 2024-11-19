import os
import re
import cv2
import json
import torch
import queue
import base64
import threading
import imagehash
import numpy as np
import concurrent.futures

from PIL import Image
from typing import List
from openai import OpenAI
from ultralytics import YOLO
from pydantic import BaseModel
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

# region_info = []
# representative_images = []

app = Flask(__name__)
yolo_model = YOLO('yolov8m.pt')
client = OpenAI(api_key="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")

FRAME_EXTRACT_DIR = './extracted_frames/'
for directory in [FRAME_EXTRACT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_frames(video_path, extract_dir, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if fps > 0 else 30

    frame_paths = []
    frame_count = 0
    saved_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                frame_filename = os.path.join(extract_dir, f"frame_{saved_count}.jpg")
                saved_count += 1
                futures.append(executor.submit(cv2.imwrite, frame_filename, frame))
                frame_paths.append(frame_filename)

            frame_count += 1

        concurrent.futures.wait(futures)

    cap.release()
    return frame_paths

def filter_frames_with_stats(frame_paths, blur_threshold=35, brightness_threshold=50):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    selected_frames = []

    deletion_stats = {
        'blurry': 0,
        'sensitive': 0,
        'dark': 0,
        'similar': 0
    }

    def filters(frame_path):
        try:
            img_pil = Image.open(frame_path)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # select_sharp_frames
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var <= blur_threshold:
                os.remove(frame_path)
                return 'blurry'

            # remove_sensitive_frames
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
            if len(faces) > 0:
                os.remove(frame_path)
                return 'sensitive'

            # select_bright_frames
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2].mean()
            if brightness <= brightness_threshold:
                os.remove(frame_path)
                return 'dark'

            return 'selected', frame_path

        except IOError:
            return None

    # filtering by workers
    selected_frame_set = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(filters, frame_path): frame_path for frame_path in frame_paths}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                continue
            status = result[0] if isinstance(result, tuple) else result
            if status == 'selected':
                selected_frame_set.add(result[1])
            elif status in deletion_stats:
                deletion_stats[status] += 1

    selected_frames = [frame_path for frame_path in frame_paths if frame_path in selected_frame_set]
    return selected_frames, deletion_stats

def convert_attributes_to_natural_language(attributes):
    attribute_descriptions = {
        'isPregnant': "is pregnant",
        'isChildren': "has children",
        'isElderly': "is elderly",
        'isDisabled': "is disabled",
        'isAllergic': "has allergies",
        'isPets': "has pets"
    }

    normalized_attributes = {key: attributes.get(key, False) if attributes.get(key) is not None else False for key in attribute_descriptions}

    if not any(normalized_attributes.values()):
        return "The user does not belong to any special group."

    population_tags = [desc for key, desc in attribute_descriptions.items() if normalized_attributes[key]]

    return "The user " + ", ".join(population_tags) + "."

def prompt_template_for_report(attributes):
    prompt_template_for_report = f"""You are a house safety assessment expert! Your task is to read image descriptions and analyze:
    1. Summarize the different regions mentioned in the description. Write one region name into 'regionName'.
    2. Identify potential safety risks in each region. Write each item into 'potentialHazards'
    3. Based on the user info, identify corresponding risks in the region. Write each item into 'specialHazards'. If the user does not belong to any special group, write 'The user does not belong to any special group.' into 'specialHazards'.
    4. Evaluate the impact of lighting and color in the region on psychological health, visibility, and comfort. Write each item into 'colorAndLightingEvaluation'.
    5. Provide suggestions for improvements based on the issues identified above. Write each item into 'suggestions'.
    6. Give a score for the region on the following criteria: personal safety, special safety, color and lighting, psychological impact, final score for this region, on a scale of 0-5. Write each item into 'scores'.

    User info: {convert_attributes_to_natural_language(attributes)}

    If multiple regions are described, repeat the analysis for each region and compile the results into the regions section of the final report. Return the report in JSON format.
    """
    return prompt_template_for_report.strip()

def prompt_template_for_frames(attributes):
    prompt_template_for_frames = f"""You are a house safety risk analyst! Your task is to analyze the following images and provide a concise but detailed evaluation covering:
    1. Provide a one-sentence brief description of which region of the house the photo shows (e.g., kitchen, bedroom, study area, dining area, etc.).
    2. Provide a detailed description of the overall room environment, including all major objects, their position, condition, and layout (e.g., furniture placement, floor cleanliness, wall decorations, or the arrangement of appliances).
    3. Identify all visible or potential safety risks (e.g., fire hazards, unstable furniture, and any factors that could pose risks).
    4. Based on the user info, identify hazards specific to that group (e.g., risks that are more accessible to or pose a greater danger to them). If the user does not belong to any special group, additional analysis can be skipped.
    5. Evaluate the room's color scheme and lighting, analyzing their effects on occupant comfort, visibility, and psychological impact.

    User info: {convert_attributes_to_natural_language(attributes)}

    All analysis should be written in one paragraph without using bullet points or separate sections.
    """
    return prompt_template_for_frames.strip()

def validate_region_data(region_data):
    required_keys = [
        'regionName',
        'potentialHazards',
        # 'specialHazards',
        'colorAndLightingEvaluation',
        'suggestions',
        'scores'
    ]
    
    # check keys
    for key in required_keys:
        if key not in region_data:
            return 0
        if not region_data[key]:
            return 0
    
    # check scores
    scores = region_data['scores']
    if len(scores) not in [4, 5]:
        return 0
    for score in scores:
        if not isinstance(score, float) or not (0 <= score <= 5):
            return 0

    return 1

def dynamic_batch_size(total_frames):
    if total_frames <= 12:
        return 5
    elif total_frames <= 30:
        return 6
    elif total_frames <= 45:
        return 7
    elif total_frames <= 75:
        return 9
    elif total_frames <= 105:
        return 11
    elif total_frames <= 165:
        return 15
    else:
        return 20

def batch_images(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

def select_representative_images_from_batches(frame_batches):
    representative_images_process = []
    
    for batch in frame_batches:
        if len(batch) > 0:
            representative_images_process.append(batch[0])
    
    return representative_images_process

def analyze_image_batch(image_paths, attributes):
    messages = []
    analysis_results = []

    messages.append({
        "role": "system",
        "content": prompt_template_for_frames(attributes)
    })

    # upload image
    for img in image_paths:
        base64_image = encode_image_base64(img)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{base64_image}", "detail": "low"
                    },
                }
            ]
        })

    # get response from api
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=275,
            stop=["\n"]
        )
    except Exception as e:
        print(f"API failed for batch: {str(e)}")
        return ""

    # get all response choices
    for choice in response.choices:
        analysis_results.append(choice.message.content)
    
    print(f"Tokens Used for the batch: {response.usage.total_tokens}")

    batch_result = '\n'.join(analysis_results)
    # print(batch_result)
    return batch_result

def process_batches(frame_batches, attributes):
    all_batch_results = [None] * len(frame_batches)

    # process batches
    print("\n-------- Batch Process START --------")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_batch_index = {
            executor.submit(analyze_image_batch, batch, attributes): batch_index
            for batch_index, batch in enumerate(frame_batches)
            if batch
        }

        for future in concurrent.futures.as_completed(future_to_batch_index):
            batch_index = future_to_batch_index[future]
            try:
                batch_result = future.result()
                if batch_result.strip():
                    all_batch_results[batch_index] = f"{batch_result}\n"
                else:
                    all_batch_results[batch_index] = ''
                print(f"Batch {batch_index + 1} processed successfully.")
            except Exception as e:
                print(f"Batch {batch_index + 1} processing failed: {str(e)}")
                all_batch_results[batch_index] = ''

    print("-------- Batch Process END, Report Process START --------")

    if not any(all_batch_results):
        print("No analysis results to summarize!")
        return [{'warning': ['nothing']}]

    combined_results = '\n'.join(result for result in all_batch_results if result)
    # print(combined_results)

    # define response & message
    class RegionEvaluation(BaseModel):
        regionName: List[str]
        potentialHazards: List[str]
        specialHazards: List[str]
        colorAndLightingEvaluation: List[str]
        suggestions: List[str]
        scores: List[float]
        
    class HouseEvaluation(BaseModel):
        regions: List[RegionEvaluation]

    messages = [
        {"role": "system", "content": prompt_template_for_report(attributes)},
        {"role": "user", "content": f"Description:\n{combined_results}"}
    ]

    # get valid response
    for retry in range(3):
        # get response from api
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=messages,
                response_format=HouseEvaluation,
            )
        except Exception as e:
            print(f"API failed for report: {str(e)}")
            print(f"Report {retry + 1} is invalid!")
            continue

        print(f"Tokens Used for the report {retry + 1}: {completion.usage.total_tokens}")
        house_evaluation = completion.choices[0].message.parsed

        # check response format
        test = house_evaluation.model_dump()
        if not (isinstance(test, dict) and test and 'regions' in test and isinstance(test['regions'], list)):
            print(f"Report {retry + 1} is invalid!")
            continue

        all_valid = True
        temp_region_info = []

        for i, region in enumerate(house_evaluation.regions):
            region_data = region.model_dump()
            validation_result = validate_region_data(region_data)
            print(f"Report {retry + 1} region {i + 1}: {validation_result}")

            if validation_result == 0:
                all_valid = False
                break
            else:
                temp_region_info.append(region_data)

        if all_valid:
            print(f"Report {retry + 1} is valid!")
            return temp_region_info
        
        print(f"Report {retry + 1} is invalid!")
        
    print("Max retries but no report with valid format!")
    return [{'warning': ['invalid']}]

def detect_and_draw_boxes(frame_paths, model, confidence_threshold=0.5):
    # detect, process and save image
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Error: Unable to load image {frame_path}")
            continue

        # YOLOv8
        detections = model(frame_path)

        # check detections
        if len(detections) == 0 or not hasattr(detections[0], 'boxes'):
            print(f"No valid detections in image: {frame_path}")
            continue

        height, width = img.shape[:2]

        # process image
        for detection in detections[0].boxes:
            if not hasattr(detection, 'xyxy') or len(detection.xyxy) == 0:
                print(f"Invalid detection data in image: {frame_path}")
                continue

            x1, y1, x2, y2 = detection.xyxy[0]
            conf = detection.conf[0]
            cls = detection.cls[0]

            # select confident class, draw box and label
            if conf > confidence_threshold:
                if int(cls) >= len(model.names):
                    print(f"Invalid class index {int(cls)} in detection for image: {frame_path}")
                    continue

                class_name = model.names[int(cls)]

                # add rectangle
                if 0 <= x1 <= width and 0 <= y1 <= height and 0 <= x2 <= width and 0 <= y2 <= height:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                else:
                    print(f"Invalid bounding box in image: {frame_path}")
                    continue

                # add label
                label = f"{class_name} {conf:.2f}"
                label_y = max(0, int(y1) - 10)
                cv2.putText(img, label, (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # save image
        cv2.imwrite(frame_path, img)

def process_video(video_path, attributes):
    print("-------------------------------- Process START --------------------------------\n")

    representative_images = []
    region_info = [{'warning': ['nothing']}]

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_frame_dir = os.path.join(FRAME_EXTRACT_DIR, f"frames_of_{video_name}")
    os.makedirs(video_frame_dir, exist_ok=True)

    # extract frames
    frame_paths = extract_frames(video_path, video_frame_dir, frame_rate=1)
    print(f"Extracted {len(frame_paths)} frames.\n")

    # delete unwanted frames
    if len(frame_paths) != 0:
        frame_paths, stats = filter_frames_with_stats(frame_paths)
        print(f"Deleted similar frames: ^^")
        print(f"Deleted blurry frames: {stats['blurry']}")
        print(f"Deleted dark frames: {stats['dark']}")
        print(f"Deleted sensitive frames: {stats['sensitive']}")
        print(f"Selected frames after filtering: {len(frame_paths)}\n")
    else:
        print("No frames extracted from the video!")
        return {"message": "No frames extracted from the video!", "regionInfo": region_info, "representativeImages": representative_images}

    # check deletions
    if len(frame_paths) == 0:
        max_deletion_step = max(stats, key=stats.get)

        if max_deletion_step == 'similar':
            region_info = [{'warning': ['similar']}]
        elif max_deletion_step == 'blurry':
            region_info = [{'warning': ['blurry']}]
        elif max_deletion_step == 'dark':
            region_info = [{'warning': ['dark']}]
        elif max_deletion_step == 'sensitive':
            region_info = [{'warning': ['sensitive']}]

        print("No valid frames left after filtering!")
        return {"message": "No valid frames left after filtering!", "regionInfo": region_info, "representativeImages": representative_images}
    
    # set batch size and batch frames
    batch_size = dynamic_batch_size(len(frame_paths))
    print(f"Dynamic batch size set to: {batch_size}")
    frame_batches = list(batch_images(frame_paths, batch_size))
    print(f"Split frames into {len(frame_batches)} batches of up to {batch_size} images each.")

    # core function
    region_info = process_batches(frame_batches, attributes)
    print("-------- Report Process END --------")

    # choose and process representative images
    representative_images = select_representative_images_from_batches(frame_batches)

    if len(representative_images) != 0 and region_info != [{'warning': ['invalid']}] and region_info != [{'warning': ['nothing']}] and region_info != []:
        try:
            print(f"\nSelected {len(representative_images)} representative images from batches.\n")
            detect_and_draw_boxes(representative_images, yolo_model)
        except Exception as e:
            print(f"YOLO failed: {str(e)}")

        # delete specialHazards
        if convert_attributes_to_natural_language(attributes) == "The user does not belong to any special group.":
            for region in region_info:
                if 'specialHazards' in region:
                    del region['specialHazards']

    return {"message": "Process END!", "regionInfo": region_info, "representativeImages": representative_images}

@app.route('/processVideo', methods=['POST'])
def process_video_endpoint():
    # video path
    print("\n")
    print("#"*80 + " START " + "#"*80)
    print("\nGET:\n")
    video_path = request.form.get('video_path')

    if video_path is None or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path!"}), 400

    print(f"Video Path:\n{video_path}\n")

    # attributes
    attributes_json = request.form.get('attributes')

    try:
        attributes = json.loads(attributes_json)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid attributes format!"}), 400

    print(f"Attributes:\n{attributes}")
    print(f"{convert_attributes_to_natural_language(attributes)}\n")

    # process video
    result = process_video(video_path, attributes)
    print("\n-------------------------------- Process END --------------------------------")
    x = result["regionInfo"]
    y = result["representativeImages"]

    # response
    print("\nRESULT:\n")
    print(f"Region Info:\n{x}\n")
    print(f"Representative Images:\n{y}\n")
    print("#"*80 + " END " + "#"*80)

    response_data = {
        "regionInfo": x,
        "representativeImages": y
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)