# Average vs Median Image Filter Comparison

This project compares the performance of the **Average Filter** and **Median Filter** for removing **salt-and-pepper noise** from digital images.

The experiment was implemented in Python using OpenCV, scikit-image, NumPy, pandas, and Matplotlib. The performance of both filters was evaluated using:

- Peak Signal-to-Noise Ratio (PSNR)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Project Objective

The main objective of this project is to determine which filtering technique performs better for removing salt-and-pepper noise while preserving image quality.

The workflow includes:

1. Loading sample images
2. Converting RGB images to grayscale
3. Adding 5% salt-and-pepper noise
4. Applying a 3×3 Average Filter
5. Applying a 3×3 Median Filter
6. Calculating PSNR, MSE, and RMSE
7. Comparing the visual and numerical results

## Images Used

Five standard images from `scikit-image` were used:

- Astronaut
- Camera
- Coins
- Moon
- Chelsea

## Technologies Used

- Python
- OpenCV
- NumPy
- pandas
- Matplotlib
- scikit-image

## Project Structure

```text
average_vs_median_image_filter_comparison/
│
├── average_vs_median_filter.py
├── README.md
│
└── outputs/
    │
    ├── images/
    │   ├── astronaut_filter_comparison.png
    │   ├── camera_filter_comparison.png
    │   ├── coins_filter_comparison.png
    │   ├── moon_filter_comparison.png
    │   └── chelsea_filter_comparison.png
    │
    └── tables/
        ├── filter_metrics.csv
        └── average_filter_performance.csv
