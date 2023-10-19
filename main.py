import os, easyocr, fasttext, time
from skimage import exposure
from skimage import io as img_io
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from orientation import level

# load character recognition model 
reader = easyocr.Reader(['fr','en'])

# load language detection model 
model = fasttext.load_model('./model/language-detection.ftz')

def preprocess_image(img):
    """ Preprocess the image to gray scale and intensifies the contrast. """
    img = rgb2gray(img)
    img = exposure.rescale_intensity(img)

    img = level(img)
    img = (img * 255).astype('uint8')

    return img 

def fix_rotation(img_path):
    """ 
    Fixing the rotation of a handwritten document 
    with OCR + Language Detection 
    """
    original = img_io.imread(img_path)

    img = preprocess_image(original)

    i = 0 
    found = False
    while i < 4: 
        # character recognition 
        img_result = reader.readtext(img)

        # language detection 
        img_lang = model.predict(
            ' '.join([item[-2] for item in img_result]) , k=1
        )

        # check if the detected language is french 
        if img_lang[0][0] == '__label__fr' and img_lang[1][0] > 0.5:
            found = True 
            break   
        
        img = np.rot90(img, k=1)
        i += 1
    
    if not found: 
        return original
    
    return np.rot90(original, k=i)

def fix_rotation_worker(input_path, output_path, filename):
    """ Worker for multithreading. """
    input_img_path = os.path.join(input_path, filename)
    fixed_img = fix_rotation(input_img_path)  
    output_img_path = os.path.join(output_path, filename)
    img_io.imsave(output_img_path, fixed_img) 
        
def fix_folder(input_path, output_path):
    """ 
    Fix the rotation of the images 
    in the folder using multiple threads
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with ThreadPoolExecutor() as executor:
        files = os.listdir(path)
        files = [ 
            f for f in files 
            if any(extension in f for extension in ['jpg', 'jpeg', 'png']) 
        ]

        start_time = time.time()  # Record start time

        for _ in tqdm(executor.map(
                    fix_rotation_worker, 
                    [input_path] * len(files),
                    [output_path] * len(files), 
                    files
                ), total=len(files) ):
            ...

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(
            f"Total Time: {elapsed_time:.2f} seconds for {len(files)} images",
            f"\nTime per Images: {elapsed_time/len(files)} seconds"
        )

if __name__ == "__main__":
    path = "./data"
    output_path = "./output"
    fix_folder(path, output_path)


