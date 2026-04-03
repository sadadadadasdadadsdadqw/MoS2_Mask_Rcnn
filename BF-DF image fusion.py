import cv2
import numpy as np
import os
import glob


def save_rgb_channels(input_path, output_folder, blue_threshold=50,
                     clip_limit=2.0, tile_grid_size=(8, 8)):
    """Separate RGB channels, enhance contrast on blue channel first, then binarize"""
    # Read color image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Unable to read image {input_path}")
        return None

    # Split BGR channels (OpenCV uses BGR order by default)
    blue_channel, green_channel, red_channel = cv2.split(img)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(input_path))[0]

    # Save original channels
    cv2.imwrite(os.path.join(output_folder, f"{filename}_red_channel.jpg"), red_channel)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_green_channel.jpg"), green_channel)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_blue_original.jpg"), blue_channel)

    # ==============================================
    # New step: Contrast enhancement on blue channel (using CLAHE)
    # ==============================================
    # Create CLAHE object (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Apply CLAHE to enhance contrast
    blue_enhanced = clahe.apply(blue_channel)
    # Save enhanced blue channel
    cv2.imwrite(os.path.join(output_folder, f"{filename}_blue_enhanced.jpg"), blue_enhanced)

    # ==============================================
    # Binarization based on enhanced blue channel
    # ==============================================
    _, blue_binary = cv2.threshold(blue_enhanced, blue_threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_blue_binary.jpg"), blue_binary)

    return blue_binary  # Return binarization result based on enhanced channel


def robust_fill_white_contours(binary_image):
    """Improved contour filling function"""
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)

    for cnt in contours:
        if cv2.contourArea(cnt) > 4:
            cv2.drawContours(filled, [cnt], -1, 255, cv2.FILLED)

    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    return filled


def blend_images(img1, img2, alpha=0.5):
    """Weighted blend of two images"""
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def subtract_images(final_results_dir, masks2_dir, output_dir):
    """Calculate difference between corresponding images in final_results and masks2"""
    os.makedirs(output_dir, exist_ok=True)

    final_results_images = glob.glob(os.path.join(final_results_dir, "*_result.png"))

    for final_result_path in final_results_images:
        filename = os.path.basename(final_result_path)
        base_name = filename.replace("_result.png", "")

        mask2_path = os.path.join(masks2_dir, f"mask_{base_name}.png")
        if not os.path.exists(mask2_path):
            print(f"Warning: Corresponding file not found in masks2 {mask2_path}")
            continue

        final_result_img = cv2.imread(final_result_path, cv2.IMREAD_GRAYSCALE)
        mask2_img = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

        if final_result_img is None or mask2_img is None:
            print(f"Error: Unable to read image {final_result_path} or {mask2_path}")
            continue

        if final_result_img.shape != mask2_img.shape:
            print(f"Resizing mask2 to match result image ({mask2_img.shape} -> {final_result_img.shape})")
            mask2_img = cv2.resize(mask2_img, (final_result_img.shape[1], final_result_img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        _, final_result_binary = cv2.threshold(final_result_img, 127, 255, cv2.THRESH_BINARY)
        _, mask2_binary = cv2.threshold(mask2_img, 127, 255, cv2.THRESH_BINARY)

        mask2_inv = cv2.bitwise_not(mask2_binary)
        subtracted_result = cv2.bitwise_and(final_result_binary, mask2_inv)

        output_path = os.path.join(output_dir, f"{base_name}_subtracted.png")
        cv2.imwrite(output_path, subtracted_result)
        print(f"√ Difference result saved: {output_path}")


def blend_subtracted_with_abc(subtracted_dir, abc_dir, output_dir):
    """Blend difference results with images from abc folder"""
    os.makedirs(output_dir, exist_ok=True)

    subtracted_images = glob.glob(os.path.join(subtracted_dir, "*_subtracted.png"))

    for subtracted_path in subtracted_images:
        filename = os.path.basename(subtracted_path)
        base_name = filename.replace("_subtracted.png", "")

        abc_image_path = os.path.join(abc_dir, f"{base_name}.jpeg")
        if not os.path.exists(abc_image_path):
            print(f"Warning: Corresponding image not found in abc folder {abc_image_path}")
            continue

        subtracted_img = cv2.imread(subtracted_path, cv2.IMREAD_GRAYSCALE)
        abc_img = cv2.imread(abc_image_path)

        if subtracted_img is None or abc_img is None:
            print(f"Error: Unable to read image {subtracted_path} or {abc_image_path}")
            continue

        if abc_img.shape[:2] != subtracted_img.shape:
            print(f"Resizing abc image to match difference result ({abc_img.shape[:2]} -> {subtracted_img.shape})")
            abc_img = cv2.resize(abc_img, (subtracted_img.shape[1], subtracted_img.shape[0]))

        subtracted_color = cv2.cvtColor(subtracted_img, cv2.COLOR_GRAY2BGR)
        blended_result = blend_images(abc_img, subtracted_color, alpha=0.7)

        output_path = os.path.join(output_dir, f"{base_name}_subtracted_overlay.jpg")
        cv2.imwrite(output_path, blended_result)
        print(f"√ Blend of difference and abc saved: {output_path}")


def process_images(input_dir, mask_dir, output_base_dir, abc_dir, blue_threshold=50,
                  clip_limit=2.0, tile_grid_size=(5, 5)):
    """Main processing function"""
    channel_dir = os.path.join(output_base_dir, "output_channels")
    filled_dir = os.path.join(output_base_dir, "filled_results")
    final_dir = os.path.join(output_base_dir, "final_results")
    subtracted_dir = os.path.join(output_base_dir, "subtracted_results")
    subtracted_overlay_dir = os.path.join(output_base_dir, "subtracted_overlay")

    os.makedirs(channel_dir, exist_ok=True)
    os.makedirs(filled_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(subtracted_dir, exist_ok=True)
    os.makedirs(subtracted_overlay_dir, exist_ok=True)

    input_images = glob.glob(os.path.join(input_dir, "*.jpeg"))
    if not input_images:
        print(f"Error: No JPEG images found in {input_dir}")
        return

    for img_path in input_images:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nProcessing image: {filename}")

        # Step 1: Split channels -> Enhance blue channel contrast -> Binarize
        print("  Separating RGB channels -> Enhancing blue channel contrast -> Binarizing...")
        blue_binary = save_rgb_channels(
            img_path, channel_dir,
            blue_threshold=blue_threshold,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size
        )
        if blue_binary is None:
            continue

        # Step 2: Fill contours
        print("  Filling white contours...")
        filled_img = robust_fill_white_contours(blue_binary)
        filled_path = os.path.join(filled_dir, f"{filename}_filled.png")
        cv2.imwrite(filled_path, filled_img)

        # Step 3: Process mask
        mask_path = os.path.join(mask_dir, f"mask_{filename}.png")
        if not os.path.exists(mask_path):
            alt_mask_path = os.path.join(mask_dir, f"{filename}_mask.png")
            if os.path.exists(alt_mask_path):
                mask_path = alt_mask_path
            else:
                print(f"Warning: Corresponding mask file not found {mask_path}")
                continue

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"Error: Unable to read mask file {mask_path}")
            continue

        if filled_img.shape != mask_img.shape:
            print(f"  Resizing mask to match image ({mask_img.shape} -> {filled_img.shape})")
            mask_img = cv2.resize(mask_img, (filled_img.shape[1], filled_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        _, binary_filled = cv2.threshold(filled_img, 127, 255, cv2.THRESH_BINARY)
        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(binary_filled, binary_mask)

        # Step 4: Subtract white parts of blue binary image
        print("  Subtracting white parts of blue binary image from result...")
        if blue_binary.shape != result.shape:
            blue_binary = cv2.resize(blue_binary, (result.shape[1], result.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        blue_binary_inv = cv2.bitwise_not(blue_binary)
        result_subtracted = cv2.bitwise_and(result, blue_binary_inv)

        result_path = os.path.join(final_dir, f"{filename}_result.png")
        cv2.imwrite(result_path, result_subtracted)
        print(f"√ Result saved: {result_path}")

    # Step 5: Calculate differences
    print("\nCalculating differences between final_results and masks2...")
    masks2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks2")
    subtract_images(final_dir, masks2_dir, subtracted_dir)

    # Step 6: Blend difference results with abc images
    print("\nBlending difference results with abc folder images...")
    blend_subtracted_with_abc(subtracted_dir, abc_dir, subtracted_overlay_dir)


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    INPUT_DIR = os.path.join(PROJECT_ROOT, "Dimage")
    MASK_DIR = os.path.join(PROJECT_ROOT, "masks")
    ABC_DIR = os.path.join(PROJECT_ROOT, "Bimage")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "process_result")

    # Adjustable parameters
    BLUE_THRESHOLD = 80  # Binarization threshold
    CLIP_LIMIT = 3.0     # CLAHE contrast limit (higher = stronger contrast, may amplify noise)
    TILE_GRID_SIZE = (8, 8)  # CLAHE tile grid size (smaller = finer detail enhancement)

    print("=" * 50)
    print("Image processing pipeline started (with blue channel contrast enhancement)")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Mask directory: {MASK_DIR}")
    print(f"ABC directory: {ABC_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing parameters: Blue threshold={BLUE_THRESHOLD}, CLAHE clip limit={CLIP_LIMIT}, tile grid size={TILE_GRID_SIZE}")
    print("=" * 50)

    process_images(
        INPUT_DIR, MASK_DIR, OUTPUT_DIR, ABC_DIR,
        blue_threshold=BLUE_THRESHOLD,
        clip_limit=CLIP_LIMIT,
        tile_grid_size=TILE_GRID_SIZE
    )
    print("\nAll image processing completed!")