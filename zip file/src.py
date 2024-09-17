import os
import torch
import easyocr  # EasyOCR library for OCR (Optical Character Recognition)
import pandas as pd
import re
import requests  # Library to make HTTP requests
from io import BytesIO
from PIL import Image  # Pillow library for image processing
from tqdm import tqdm  # Import tqdm for progress bar functionality

# Define a mapping for short forms of units to their full names
short_form_map = {
    'g': 'gram',
    'kg': 'kilogram',
    'mg': 'milligram',
    'lb': 'pound',
    'oz': 'ounce',
    'cm': 'centimetre',
    'mm': 'millimetre',
    'm': 'metre',
    'ft': 'foot',
    'in': 'inch',
    'yd': 'yard',
    'kv': 'kilovolt',
    'v': 'volt',
    'w': 'watt',
    'kw': 'kilowatt',
    'ml': 'millilitre',
    'l': 'litre',
    'cl': 'centilitre',
    'fl oz': 'fluid ounce',
    'pt': 'pint',
    'qt': 'quart',
    'gal': 'gallon'
}

# Define the ImageEntityModel class for extracting and processing information from images
class ImageEntityModel:

    def __init__(self, device=None):
        # Set the device to either 'cuda' (GPU) if available or 'cpu'
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize EasyOCR model for reading English text
        self.reader = easyocr.Reader(['en'])
        
        # Define a mapping of entity types to valid units
        self.entity_unit_map = {
            'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'item_weight': {'gram', 'kilogram', 'milligram', 'ounce', 'pound'},
            'maximum_weight_recommendation': {'gram', 'kilogram', 'milligram', 'ounce', 'pound'},
            'voltage': {'kilovolt', 'volt'},
            'wattage': {'kilowatt', 'watt'},
            'item_volume': {'millilitre', 'litre', 'fluid ounce', 'pint', 'gallon', 'quart'}
        }

    def normalize_short_forms(self, extracted_text):
        """Normalize short forms of units found in the OCR-extracted text."""
        words = extracted_text.split()  # Split text into words
        # Replace short forms with their full names
        normalized_words = [short_form_map.get(word.lower(), word) for word in words]
        return " ".join(normalized_words)  # Return the normalized text

    def extract_ocr_text(self, image_path):
        """Use EasyOCR to extract text from an image file."""
        try:
            # Perform OCR on the image and join the results into a single string
            ocr_results = self.reader.readtext(image_path, detail=0)
            extracted_text = " ".join(ocr_results)
            # Normalize any short forms of units
            normalized_text = self.normalize_short_forms(extracted_text)
            return normalized_text
        except Exception as e:
            # Print an error message if OCR fails
            print(f"Error performing OCR on {image_path}: {e}")
            return None

    def match_entity_to_ocr(self, extracted_text):
        """Match extracted OCR text to possible entities using predefined units."""
        if extracted_text is None:
            return None

        # Initialize a dictionary to store matches for each entity
        matched_entities = {key: [] for key in self.entity_unit_map.keys()}

        # Regex pattern to identify measurements (number + unit)
        pattern = r'(\d+\.?\d*)\s*(' + '|'.join(re.escape(unit) for unit in short_form_map.values()) + r')'
        matches = re.findall(pattern, extracted_text.lower())  # Find all matches

        # For each match, check if the unit is valid for any entity type
        for match in matches:
            value, unit = match
            for entity_name, units in self.entity_unit_map.items():
                if unit in units:
                    matched_entities[entity_name].append(f"{value} {unit}")

        return matched_entities  # Return the matched entities

    def process_dimensions(self, dimensions):
        """Process and assign dimension values to height, width, and depth."""
        dimension_values = []
        units = []

        # Extract dimension values and units from the input list
        for dim in dimensions:
            match = re.match(r'(\d+\.?\d*)\s*(\w+)', dim)
            if match:
                value, unit = match.groups()
                dimension_values.append(float(value))
                units.append(unit)

        # If no valid dimensions are found, return empty values
        if len(dimension_values) == 0:
            return {"height": "", "width": "", "depth": ""}

        # If only one dimension is found, use it for all three (height, width, depth)
        if len(dimension_values) == 1:
            return {"height": f"{dimension_values[0]} {units[0]}", 
                    "width": f"{dimension_values[0]} {units[0]}", 
                    "depth": f"{dimension_values[0]} {units[0]}"}

        # If two dimensions are found, assign them to height and width
        if len(dimension_values) == 2:
            return {"height": f"{max(dimension_values)} {units[0]}",
                    "width": f"{min(dimension_values)} {units[1]}",
                    "depth": ""}

        # If all three dimensions are found, assign them accordingly
        return {"height": f"{dimension_values[0]} {units[0]}",
                "width": f"{dimension_values[1]} {units[1]}",
                "depth": f"{dimension_values[2]} {units[2]}"}

    def process_weights(self, weights):
        """Process and assign weight values to item_weight and maximum_weight_recommendation."""
        weight_values = []

        # Extract weight values from the input list
        for weight in weights:
            match = re.match(r'(\d+\.?\d*)\s*(\w+)', weight)
            if match:
                value = float(match.groups()[0])
                weight_values.append(value)

        # If no valid weights are found, return empty values
        if len(weight_values) == 0:
            return {"item_weight": "", "maximum_weight_recommendation": ""}

        # If only one weight is found, assign it to both item_weight and maximum_weight_recommendation
        if len(weight_values) == 1:
            return {"item_weight": f"{weight_values[0]} kilogram",
                    "maximum_weight_recommendation": f"{weight_values[0]} kilogram"}

        # If multiple weights are found, assign the smallest to item_weight and largest to maximum_weight_recommendation
        return {"item_weight": f"{min(weight_values)} kilogram",
                "maximum_weight_recommendation": f"{max(weight_values)} kilogram"}

    def predict(self, image_link, category_id, entity_name):
        """Process the image from the given link and extract entity values."""
        try:
            # Download the image from the link
            response = requests.get(image_link)
            img = Image.open(BytesIO(response.content))
            img_path = "temp_image.jpg"
            img.save(img_path)  # Save the image temporarily for OCR

            # Perform OCR to extract text from the image
            extracted_text = self.extract_ocr_text(img_path)

            # Debugging: Print the extracted text
            print(f"Extracted text from image: {extracted_text}")

            # Match the extracted text to known entity types
            matched_entities = self.match_entity_to_ocr(extracted_text)

            # If no entities are matched, return an empty string
            if matched_entities is None:
                return ""

            # If the entity is a dimension (height, width, depth), process it
            if entity_name in ['height', 'width', 'depth']:
                dimensions = [value for key in ['height', 'width', 'depth'] for value in matched_entities[key]]
                processed_dimensions = self.process_dimensions(dimensions)
                return processed_dimensions.get(entity_name, "")

            # If the entity is a weight, process it
            if entity_name in ['item_weight', 'maximum_weight_recommendation']:
                weights = matched_entities['item_weight'] + matched_entities['maximum_weight_recommendation']
                processed_weights = self.process_weights(weights)
                return processed_weights.get(entity_name, "")

            # If any other entity, return the first matched value
            if matched_entities.get(entity_name):
                return matched_entities[entity_name][0]

            return ""  # Return empty string if no valid match is found

        except Exception as e:
            # Print error if any exception occurs during the process
            print(f"Error processing image from {image_link}: {e}")
            return ""

# Instantiate the ImageEntityModel class
image_entity_model = ImageEntityModel()

# Define the predictor function to call the model and get the entity value
def predictor(image_link, category_id, entity_name):
    """
    Call the ImageEntityModel to extract entity value from the image.
    This function is called for each test case.
    """
    return image_entity_model.predict(image_link, category_id, entity_name)

# Process the test data
if __name__ == "__main__":
    # Path to the test CSV file
    test_file = "test.csv"
    
    # Load the test dataset
    test_df = pd.read_csv(test_file)
    
    # Initialize an empty list to store the predictions
    predictions = []

    # Iterate over the test dataset and make predictions for each row
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_link = row['image_link']  # Extract image link
        category_id = row['group_id']  # Extract category ID
        entity_name = row['entity_name']  # Extract entity name

        # Call the predictor function to get the prediction
        prediction = predictor(image_link, category_id, entity_name)

        # Append the result (index and prediction) to the predictions list
        predictions.append({'index': row['index'], 'prediction': prediction})

    # Convert the predictions list into a DataFrame
    prediction_df = pd.DataFrame(predictions)

    # Save the predictions to a CSV file
    prediction_df.to_csv('test_out.csv', index=False)

    # Debugging: Print the first few rows of the output CSV
    print(prediction_df.head())

