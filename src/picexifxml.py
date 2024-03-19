import os
from PIL import Image, ExifTags
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Przygotowanie modelu i toknizera
model_id = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
feature_extractor = ViTImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Parametry generowania opisów
max_length = 1000
num_beams = 5
no_repeat_ngram_size = 2
early_stopping = True
gen_kwargs = {
    "max_length": max_length,
    "num_beams": num_beams,
    "no_repeat_ngram_size": no_repeat_ngram_size,
    "early_stopping": early_stopping
}


def predict_step(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()


def generate_xml_for_image(image_path, image_id, xml_directory):
    exif_data = {}
    try:
        with Image.open(image_path) as image:
            exif_data = {ExifTags.TAGS.get(k, k): v for k, v in image._getexif().items() if k in ExifTags.TAGS}
    except Exception as e:
        print(f"Nie można odczytać danych EXIF dla {image_path}: {e}")

    photo_elem = ET.Element("photo", id=str(image_id), name=os.path.basename(image_path))
    ET.SubElement(photo_elem, "EXIF").text = str(exif_data)
    ET.SubElement(photo_elem, "description").text = predict_step(image_path)

    xml_str = ET.tostring(photo_elem, encoding="utf-8")
    pretty_xml_as_string = minidom.parseString(xml_str).toprettyxml(indent="   ")

    os.makedirs(xml_directory, exist_ok=True)
    xml_path = os.path.join(xml_directory, os.path.basename(image_path) + '.xml')
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_string)


def process_directory(directory_path, xml_directory):
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(".jpg")]
    total_files = len(files)

    # Wyświetlanie paska postępu na starcie z 0%
    progressBar(total_files, iteration=0, prefix='Progress:', suffix='Complete', length=65)

    for i, filename in enumerate(files):
        generate_xml_for_image(os.path.join(directory_path, filename), i + 1, xml_directory)
        progressBar(total_files, iteration=i + 1, prefix='Progress:', suffix='Complete', length=65)

    print()


def progressBar(total, iteration, prefix='', suffix='', decimals=1, length=100, fill='▓', printEnd="\n"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print(f'{prefix} |{bar}| {percent}% {suffix}', end=printEnd)


if __name__ == "__main__":
    images_directory = 'images'
    xml_directory = 'generated_xml'
    process_directory(images_directory, xml_directory)
