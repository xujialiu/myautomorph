import fundus_prep as prep
import os
import pandas as pd
from PIL import ImageFile
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import warnings
warnings.filterwarnings('ignore')

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA', '..')

def process_single_image(args):
    """Process a single image"""
    image_path, save_path, resolution_dict = args
    
    # Skip if already processed
    if os.path.exists(f'{AUTOMORPH_DATA}/Results/M0/images/' + image_path):
        return None
    
    try:
        dst_image = f'{AUTOMORPH_DATA}/images/' + image_path
        
        # Get resolution
        resolution_ = resolution_dict.get(image_path, None)
        if resolution_ is None:
            return None
            
        # Process image
        img = prep.imread(dst_image)
        radius_list = []
        centre_list_w = []
        centre_list_h = []
        
        r_img, borders, mask, r_img, radius_list, centre_list_w, centre_list_h = prep.process_without_gb(
            img, img, radius_list, centre_list_w, centre_list_h
        )
        
        # Save processed image
        output_name = image_path.split('.')[0] + '.png'
        prep.imwrite(save_path + output_name, r_img)
        
        # Calculate scale
        radius = radius_list[0] if radius_list else 0
        scale = radius * 2 / 912
        scale_resolution = resolution_ * scale * 1000
        
        return {
            'Name': output_name,
            'centre_w': centre_list_w[0] if centre_list_w else 0,
            'centre_h': centre_list_h[0] if centre_list_h else 0,
            'radius': radius,
            'Scale': scale,
            'Scale_resolution': scale_resolution
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_batch(batch_args):
    """Process a batch of images"""
    results = []
    for args in batch_args:
        result = process_single_image(args)
        if result is not None:
            results.append(result)
    return results

def process_multiprocessing(image_list, save_path, num_workers=None):
    """Main multiprocessing function"""
    
    # Load resolution information
    resolution_df = pd.read_csv(f'{AUTOMORPH_DATA}/resolution_information.csv')
    resolution_dict = dict(zip(resolution_df['fundus'], resolution_df['res']))
    
    # Prepare arguments for each image
    args_list = [(img_path, save_path, resolution_dict) for img_path in image_list]
    
    # Filter out already processed images
    args_list = [args for args in args_list 
                 if not os.path.exists(f'{AUTOMORPH_DATA}/Results/M0/images/' + args[0])]
    
    if not args_list:
        print("All images already processed!")
        return
    
    # Set number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Use max 8 cores to avoid memory issues
    
    print(f"Processing {len(args_list)} images using {num_workers} workers...")
    
    # Split work into batches for better progress tracking
    batch_size = max(1, len(args_list) // (num_workers * 4))
    batches = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]
    
    # Process with multiprocessing and progress bar
    results = []
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        with tqdm(total=len(args_list), desc="Processing images") as pbar:
            for batch_result in pool.imap_unordered(process_batch, batches):
                results.extend(batch_result)
                pbar.update(len(batch_result))
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f'{AUTOMORPH_DATA}/Results/M0/crop_info.csv', index=None, encoding='utf8')
        print(f"Successfully processed {len(results)} images")
    else:
        print("No images were successfully processed")

def process_sequential(image_list, save_path):
    """Sequential processing with progress bar (fallback option)"""
    
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []
    
    resolution_list = pd.read_csv(f'{AUTOMORPH_DATA}/resolution_information.csv')
    
    for image_path in tqdm(image_list, desc="Processing images"):
        
        dst_image = f'{AUTOMORPH_DATA}/images/' + image_path
        if os.path.exists(f'{AUTOMORPH_DATA}/Results/M0/images/' + image_path):
            continue
            
        try:
            resolution_ = resolution_list['res'][resolution_list['fundus']==image_path].values[0]
            list_resolution.append(resolution_)
            img = prep.imread(dst_image)
            r_img, borders, mask, r_img, radius_list, centre_list_w, centre_list_h = prep.process_without_gb(
                img, img, radius_list, centre_list_w, centre_list_h
            )
            prep.imwrite(save_path + image_path.split('.')[0] + '.png', r_img)
            name_list.append(image_path.split('.')[0] + '.png')
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    if name_list:
        scale_list = [a*2/912 for a in radius_list]
        scale_resolution = [a*b*1000 for a,b in zip(list_resolution, scale_list)]
        Data4stage2 = pd.DataFrame({
            'Name': name_list, 
            'centre_w': centre_list_w, 
            'centre_h': centre_list_h, 
            'radius': radius_list, 
            'Scale': scale_list, 
            'Scale_resolution': scale_resolution
        })
        Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M0/crop_info.csv', index=None, encoding='utf8')

if __name__ == "__main__":
    # Clean up hidden files
    if os.path.exists(f'{AUTOMORPH_DATA}/images/.ipynb_checkpoints'):
        shutil.rmtree(f'{AUTOMORPH_DATA}/images/.ipynb_checkpoints')
    
    # Get image list
    image_list = sorted(os.listdir(f'{AUTOMORPH_DATA}/images'))
    save_path = f'{AUTOMORPH_DATA}/Results/M0/images/'
    
    # Create output directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Choose processing method
    use_multiprocessing = True  # Set to False to use sequential processing
    num_workers = None
    
    if use_multiprocessing:
        # Multiprocessing with configurable number of workers
          # Set to None for auto-detection, or specify a number
        process_multiprocessing(image_list, save_path, num_workers)
    else:
        # Sequential processing (original method)
        process_sequential(image_list, save_path)