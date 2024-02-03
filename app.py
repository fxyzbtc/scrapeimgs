from PIL import Image
from pillow_heif import register_heif_opener
from transformers import pipeline
from datetime import datetime
from loguru import logger
import sys
import os
from tqdm import tqdm
from random import choice
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from PIL import ImageFile
register_heif_opener()

# change the loger level to info
logger.remove()
# logger.add(sys.stdout, level="INFO")
logger.add(sys.stdout, level="WARNING")
logger.add(sys.stderr, level="ERROR")


# init models
## translation
# Use a pipeline as a high-level helper

translation = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

## image captioning
ImageFile.LOAD_TRUNCATED_IMAGES = True
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_file_modification_time(fp: Path):
    '''extract the modification time and return the YYYYMMDD format'''
    return datetime.fromtimestamp(fp.stat().st_mtime).strftime('%Y%m%d')

def captioning(img_url):
    '''captioning the image and return the title'''
    raw_image = Image.open(img_url).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    title_en = blip_processor.decode(out[0], skip_special_tokens=True)

    # translation
    title_cn = translation(title_en, max_length=100)[0]['translation_text']
    return title_cn + "_" + "_".join(title_en.split(" ")) 

def captioning_all_in_path(fp: Path):
    '''captioning all the images in the path'''
    for img_fp in tqdm(fp.glob('*')):
        if len(img_fp.name) > 30:
            logger.debug(f"✅|SKIP| File {img_fp} __len__ > 30")
            continue
        if len(img_fp.name.split('_')) >= 4:
            logger.debug(f"✅|SKIP| File {img_fp} having too many _ in the name, seems to be captioned already")
            continue

        ext = img_fp.suffix
        parent = img_fp.parent
        try:
            new_img_fp = captioning(img_fp)
            dt = extract_file_modification_time(img_fp)
            new_img_fp = parent / (dt +"_" + new_img_fp + ext)

        except Image.UnidentifiedImageError:
            logger.warning(f"✅| Not Implemented| Image {img_fp}")
            new_img_fp = parent / (extract_file_modification_time(img_fp) + "_" + img_fp.name)
        except Exception as e:
            logger.warning(f"❌|ERROR | File {img_fp} reported error {e}")
            continue
        
        try:
            os.rename(img_fp, new_img_fp)
            logger.info(f"✅ | DONE| File {img_fp} renamed to {new_img_fp}")
        except FileExistsError:
            logger.warning(f"✅|File {new_img_fp} already exists, trying to rename it with alternative name with random number")
            new_img_fp = parent / (dt + "_" + str(choice(range(100))) + "_" + new_img_fp.name + ext)
            try:
                os.rename(img_fp, new_img_fp)
            except Exception as e:
                logger.warning(f"❌|ERROR | File {img_fp} reported error {e}")
        except Exception as e:
            logger.warning(f"❌|ERROR | File {img_fp} reported error {e}")

if __name__ == '__main__':
    '''interate all the images in the path and sub-path and captioning them'''
    fp = Path(sys.argv[1])
    captioning_all_in_path(fp)
    logger.info("🎉|DONE| All the images in the path and sub-path are captioned")