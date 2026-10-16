import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import data, color
from skimage.util import random_noise
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


# ============================================================
# CONFIGURATION
# ============================================================

NOISE_AMOUNT = 0.05
AVERAGE_KERNEL_SIZE = (3, 3)
MEDIAN_KERNEL_SIZE = 3
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# Output folders
OUTPUT_DIR = Path("outputs")
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
TABLE_OUTPUT_DIR = OUTPUT_DIR / "tables"

IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: LOAD SAMPLE IMAGES
# ============================================================

images = {
    "Astronaut": data.astronaut(),
    "Camera": data.camera(),
    "Coins": data.coins(),
    "Moon": data.moon(),
    "Chelsea": data.chelsea()
}

print("Images loaded successfully!")


# ============================================================
# STEP 2: CONVERT TO GRAYSCALE AND ADD SALT-AND-PEPPER NOISE
# ============================================================

processed_images = {}

for name, image in images.items():

    # Convert RGB images to grayscale
    if image.ndim == 3:
        gray = color.rgb2gray(image)

        # Convert grayscale image from [0, 1] to [0, 255]
        gray = (gray * 255).astype(np.uint8)

    else:
        gray = image.astype(np.uint8)

    # Add 5% salt-and-pepper noise
    noisy = random_noise(
        gray,
        mode="s&p",
        amount=NOISE_AMOUNT,
        rng=rng
    )

    # Convert noisy image from [0, 1] to [0, 255]
    noisy = (noisy * 255).astype(np.uint8)

    processed_images[name] = {
        "original": gray,
        "noisy": noisy
    }

    print(
        f"{name}: "
        f"Original shape = {gray.shape}, "
        f"Noisy shape = {noisy.shape}"
    )

print("\nSalt-and-pepper noise added successfully!")


# ============================================================
# STEP 3: APPLY AVERAGE FILTER AND MEDIAN FILTER
# ============================================================

for name, img_data in processed_images.items():

    noisy = img_data["noisy"]

    # Apply 3x3 Average Filter
    average_filtered = cv2.blur(
        noisy,
        AVERAGE_KERNEL_SIZE
    )

    # Apply 3x3 Median Filter
    median_filtered = cv2.medianBlur(
        noisy,
        MEDIAN_KERNEL_SIZE
    )

    # Store filtered images
    processed_images[name]["average"] = average_filtered
    processed_images[name]["median"] = median_filtered

    print(
        f"{name}: Average and Median filters applied successfully!"
    )

print("\nAll filters applied successfully!")


# ============================================================
# STEP 4: CALCULATE PSNR, MSE, AND RMSE
# ============================================================

results = []

for name, img_data in processed_images.items():

    original = img_data["original"]
    average_filtered = img_data["average"]
    median_filtered = img_data["median"]


    average_mse = mean_squared_error(
        original,
        average_filtered
    )

    average_rmse = np.sqrt(average_mse)

    average_psnr = peak_signal_noise_ratio(
        original,
        average_filtered,
        data_range=255
    )


    median_mse = mean_squared_error(
        original,
        median_filtered
    )

    median_rmse = np.sqrt(median_mse)

    median_psnr = peak_signal_noise_ratio(
        original,
        median_filtered,
        data_range=255
    )


    results.append({
        "Image": name,
        "Filter": "Average Filter",
        "PSNR_dB": average_psnr,
        "MSE": average_mse,
        "RMSE": average_rmse
    })

    results.append({
        "Image": name,
        "Filter": "Median Filter",
        "PSNR_dB": median_psnr,
        "MSE": median_mse,
        "RMSE": median_rmse
    })


    print("\n" + "=" * 65)
    print(f"IMAGE: {name}")
    print("=" * 65)

    print("\nAverage Filter")
    print("-" * 30)
    print(f"PSNR : {average_psnr:.4f} dB")
    print(f"MSE  : {average_mse:.4f}")
    print(f"RMSE : {average_rmse:.4f}")

    print("\nMedian Filter")
    print("-" * 30)
    print(f"PSNR : {median_psnr:.4f} dB")
    print(f"MSE  : {median_mse:.4f}")
    print(f"RMSE : {median_rmse:.4f}")


# ============================================================
# STEP 5: CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "PSNR_dB": lambda x: f"{x:.4f}",
            "MSE": lambda x: f"{x:.4f}",
            "RMSE": lambda x: f"{x:.4f}"
        }
    )
)


# ============================================================
# STEP 6: SAVE DETAILED RESULTS AS CSV
# ============================================================

metrics_file = TABLE_OUTPUT_DIR / "filter_metrics.csv"

results_df.to_csv(
    metrics_file,
    index=False
)

print(
    f"\nDetailed performance results saved to: {metrics_file}"
)


# ============================================================
# STEP 7: CALCULATE AVERAGE PERFORMANCE
# ============================================================

average_performance = (
    results_df
    .groupby("Filter")[["PSNR_dB", "MSE", "RMSE"]]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 80)
print("AVERAGE FILTER PERFORMANCE")
print("=" * 80)

print(
    average_performance.to_string(
        index=False,
        formatters={
            "PSNR_dB": lambda x: f"{x:.4f}",
            "MSE": lambda x: f"{x:.4f}",
            "RMSE": lambda x: f"{x:.4f}"
        }
    )
)


# ============================================================
# STEP 8: SAVE AVERAGE PERFORMANCE AS CSV
# ============================================================

average_file = TABLE_OUTPUT_DIR / "average_filter_performance.csv"

average_performance.to_csv(
    average_file,
    index=False
)

print(
    f"\nAverage performance results saved to: {average_file}"
)


# ============================================================
# STEP 9: VISUALIZE AND SAVE RESULTS
# ============================================================

for name, img_data in processed_images.items():

    original = img_data["original"]
    noisy = img_data["noisy"]
    average_filtered = img_data["average"]
    median_filtered = img_data["median"]

    plt.figure(figsize=(16, 5))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Noisy image
    plt.subplot(1, 4, 2)
    plt.imshow(noisy, cmap="gray")
    plt.title("Salt & Pepper Noise")
    plt.axis("off")

    # Average filtered image
    plt.subplot(1, 4, 3)
    plt.imshow(average_filtered, cmap="gray")
    plt.title("Average Filter")
    plt.axis("off")

    # Median filtered image
    plt.subplot(1, 4, 4)
    plt.imshow(median_filtered, cmap="gray")
    plt.title("Median Filter")
    plt.axis("off")

    plt.suptitle(
        f"{name} - Average vs Median Filter",
        fontsize=16
    )

    plt.tight_layout()

    # Save figure
    filename = (
        IMAGE_OUTPUT_DIR /
        f"{name.lower()}_filter_comparison.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved comparison image: {filename}")

    plt.show()
    plt.close()


# ============================================================
# STEP 10: PRINT FINAL WINNER
# ============================================================

average_filter_row = average_performance[
    average_performance["Filter"] == "Average Filter"
].iloc[0]

median_filter_row = average_performance[
    average_performance["Filter"] == "Median Filter"
].iloc[0]

print("\n")
print("=" * 80)
print("OVERALL COMPARISON")
print("=" * 80)

print(
    f"Average Filter Mean PSNR : "
    f"{average_filter_row['PSNR_dB']:.4f} dB"
)

print(
    f"Median Filter Mean PSNR  : "
    f"{median_filter_row['PSNR_dB']:.4f} dB"
)

print(
    f"Average Filter Mean MSE  : "
    f"{average_filter_row['MSE']:.4f}"
)

print(
    f"Median Filter Mean MSE   : "
    f"{median_filter_row['MSE']:.4f}"
)

print(
    f"Average Filter Mean RMSE : "
    f"{average_filter_row['RMSE']:.4f}"
)

print(
    f"Median Filter Mean RMSE  : "
    f"{median_filter_row['RMSE']:.4f}"
)


if (
    median_filter_row["PSNR_dB"] > average_filter_row["PSNR_dB"]
    and median_filter_row["MSE"] < average_filter_row["MSE"]
    and median_filter_row["RMSE"] < average_filter_row["RMSE"]
):
    print(
        "\nOverall Result: "
        "Median Filter performed better than Average Filter."
    )

elif (
    average_filter_row["PSNR_dB"] > median_filter_row["PSNR_dB"]
    and average_filter_row["MSE"] < median_filter_row["MSE"]
    and average_filter_row["RMSE"] < median_filter_row["RMSE"]
):
    print(
        "\nOverall Result: "
        "Average Filter performed better than Median Filter."
    )

else:
    print(
        "\nOverall Result: "
        "The filters produced mixed performance across the metrics."
    )


print("\n" + "=" * 80)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 80)