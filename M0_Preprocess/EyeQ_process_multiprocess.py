import fundus_prep as prep
import os
import pandas as pd
import shutil

from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import warnings
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")

AUTOMORPH_DATA = os.getenv("AUTOMORPH_DATA", "..")


def process_single_image(args):
    """Process a single image"""
    image_path, save_path, resolution_dict = args

    # Skip if already processed
    if os.path.exists(f"{AUTOMORPH_DATA}/Results/M0/images/" + image_path):
        return None

    try:
        dst_image = f"{AUTOMORPH_DATA}/images/" + image_path

        # Get resolution
        resolution_ = resolution_dict.get(image_path, None)
        if resolution_ is None:
            return None

        # Process image
        img = prep.imread(dst_image)
        radius_list = []
        centre_list_w = []
        centre_list_h = []

        r_img, borders, mask, r_img, radius_list, centre_list_w, centre_list_h = (
            prep.process_without_gb(img, img, radius_list, centre_list_w, centre_list_h)
        )

        # Save processed image
        output_name = image_path.split(".")[0] + ".png"
        prep.imwrite(save_path + output_name, r_img)

        # Calculate scale
        radius = radius_list[0] if radius_list else 0
        scale = radius * 2 / 912
        scale_resolution = resolution_ * scale * 1000

        return {
            "Name": output_name,
            "centre_w": centre_list_w[0] if centre_list_w else 0,
            "centre_h": centre_list_h[0] if centre_list_h else 0,
            "radius": radius,
            "Scale": scale,
            "Scale_resolution": scale_resolution,
        }
    except Exception as e:
        print(f"\nError processing {image_path}: {str(e)}")
        return None


def init_worker(resolution_dict_):
    """Initialize worker process with shared data"""
    global resolution_dict
    resolution_dict = resolution_dict_


def process_single_image_worker(image_path_and_save_path):
    """Worker function that uses global resolution_dict"""
    image_path, save_path = image_path_and_save_path
    return process_single_image((image_path, save_path, resolution_dict))


def process_multiprocessing(image_list, save_path, num_workers=None):
    """Main multiprocessing function with proper progress bar"""

    # Load resolution information
    resolution_df = pd.read_csv(f"{AUTOMORPH_DATA}/resolution_information.csv")
    resolution_dict = dict(zip(resolution_df["fundus"], resolution_df["res"]))

    # Filter out already processed images
    images_to_process = [
        img
        for img in image_list
        if not os.path.exists(f"{AUTOMORPH_DATA}/Results/M0/images/" + img)
    ]

    if not images_to_process:
        print("All images already processed!")
        return

    # Set number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Use max 8 cores to avoid memory issues

    print(f"Processing {len(images_to_process)} images using {num_workers} workers...")

    # Prepare arguments
    args_list = [(img, save_path) for img in images_to_process]

    # Process with multiprocessing and progress bar
    results = []

    # Method 1: Using imap with individual progress updates
    with Pool(
        processes=num_workers, initializer=init_worker, initargs=(resolution_dict,)
    ) as pool:
        with tqdm(total=len(args_list), desc="Processing images", unit="img") as pbar:
            for result in pool.imap_unordered(process_single_image_worker, args_list):
                if result is not None:
                    results.append(result)
                pbar.update(1)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(
            f"{AUTOMORPH_DATA}/Results/M0/crop_info.csv", index=None, encoding="utf8"
        )
        print(f"\nSuccessfully processed {len(results)} images")
    else:
        print("\nNo images were successfully processed")


def process_multiprocessing_alternative(image_list, save_path, num_workers=None):
    """Alternative multiprocessing with map and chunksize"""

    # Load resolution information
    resolution_df = pd.read_csv(f"{AUTOMORPH_DATA}/resolution_information.csv")
    resolution_dict = dict(zip(resolution_df["fundus"], resolution_df["res"]))

    # Filter out already processed images
    images_to_process = [
        img
        for img in image_list
        if not os.path.exists(f"{AUTOMORPH_DATA}/Results/M0/images/" + img)
    ]

    if not images_to_process:
        print("All images already processed!")
        return

    # Set number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    print(f"Processing {len(images_to_process)} images using {num_workers} workers...")

    # Prepare arguments
    args_list = [(img, save_path, resolution_dict) for img in images_to_process]

    # Calculate optimal chunksize
    chunksize = max(1, len(args_list) // (num_workers * 4))

    # Process with multiprocessing
    with Pool(processes=num_workers) as pool:
        results_iter = pool.imap(process_single_image, args_list, chunksize=chunksize)
        results = []

        # Manual progress bar update
        with tqdm(total=len(args_list), desc="Processing images", unit="img") as pbar:
            for result in results_iter:
                if result is not None:
                    results.append(result)
                pbar.update(1)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(
            f"{AUTOMORPH_DATA}/Results/M0/crop_info.csv", index=None, encoding="utf8"
        )
        print(f"\nSuccessfully processed {len(results)} images")
    else:
        print("\nNo images were successfully processed")


def process_sequential(image_list, save_path):
    """Sequential processing with progress bar (fallback option)"""

    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []

    resolution_list = pd.read_csv(f"{AUTOMORPH_DATA}/resolution_information.csv")

    for image_path in tqdm(image_list, desc="Processing images", unit="img"):
        dst_image = f"{AUTOMORPH_DATA}/images/" + image_path
        if os.path.exists(f"{AUTOMORPH_DATA}/Results/M0/images/" + image_path):
            continue

        try:
            resolution_ = resolution_list["res"][
                resolution_list["fundus"] == image_path
            ].values[0]
            list_resolution.append(resolution_)
            img = prep.imread(dst_image)
            r_img, borders, mask, r_img, radius_list, centre_list_w, centre_list_h = (
                prep.process_without_gb(
                    img, img, radius_list, centre_list_w, centre_list_h
                )
            )
            prep.imwrite(save_path + image_path.split(".")[0] + ".png", r_img)
            name_list.append(image_path.split(".")[0] + ".png")
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")

    if name_list:
        scale_list = [a * 2 / 912 for a in radius_list]
        scale_resolution = [a * b * 1000 for a, b in zip(list_resolution, scale_list)]
        Data4stage2 = pd.DataFrame(
            {
                "Name": name_list,
                "centre_w": centre_list_w,
                "centre_h": centre_list_h,
                "radius": radius_list,
                "Scale": scale_list,
                "Scale_resolution": scale_resolution,
            }
        )
        Data4stage2.to_csv(
            f"{AUTOMORPH_DATA}/Results/M0/crop_info.csv", index=None, encoding="utf8"
        )
        print(f"\nSuccessfully processed {len(name_list)} images")


if __name__ == "__main__":
    # Clean up hidden files
    if os.path.exists(f"{AUTOMORPH_DATA}/images/.ipynb_checkpoints"):
        shutil.rmtree(f"{AUTOMORPH_DATA}/images/.ipynb_checkpoints")

    # Get image list
    image_list = sorted(os.listdir(f"{AUTOMORPH_DATA}/images"))
    save_path = f"{AUTOMORPH_DATA}/Results/M0/images/"

    # Create output directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Choose processing method
    use_multiprocessing = True  # Set to False to use sequential processing
    num_workers = 32  # Set to None for auto-detection, or specify a number

    if use_multiprocessing:
        # Use the main method (with imap_unordered for better performance)
        process_multiprocessing(image_list, save_path, num_workers)

        # Or use alternative method (with imap for ordered results)
        # process_multiprocessing_alternative(image_list, save_path, num_workers)
    else:
        # Sequential processing (original method)
        process_sequential(image_list, save_path)
