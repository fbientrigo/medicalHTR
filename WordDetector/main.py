import argparse
from typing import List

import cv2
import matplotlib.pyplot as plt
from path import Path

from word_detector import detect, prepare_img, sort_multiline
from filer import init_folder, find_matching_file

def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../data/line'))
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--base_height', type=int, default=500, help='Base height to normalize the image size')
    parser.add_argument('--sigma_x', type=float, help='Optional custom sigma for x-direction')
    parser.add_argument('--sigma_y', type=float, help='Optional custom sigma for y-direction')
    parsed = parser.parse_args()

    # Usar init_folder y find_matching_file para obtener las imágenes
    datasets_path, files = init_folder()
    matching_files = find_matching_file(datasets_path, files)

    img_files = [Path(datasets_path) / file for file in matching_files]

    for fn_img in img_files:
        print(f'Processing file {fn_img}')

        # load image and process it
        img = cv2.imread(str(fn_img))
        height, width = img.shape[:2]
        scale = parsed.base_height / height
        new_width = int(width * scale)
        img = cv2.resize(img, (new_width, parsed.base_height))

        img = prepare_img(img, parsed.base_height)

        # Use sigma_x and sigma_y if provided, otherwise use defaults
        sigma_x = parsed.sigma_x if parsed.sigma_x is not None else parsed.sigma * parsed.theta
        sigma_y = parsed.sigma_y if parsed.sigma_y is not None else parsed.sigma
        print(sigma_x, sigma_y)
        detections = detect(img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma_x,
                            theta=parsed.sigma_y,
                            min_area=parsed.min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)

        # plot results
        plt.imshow(img, cmap='gray')
        num_colors = 7
        colors = plt.cm.get_cmap('rainbow', num_colors)

        

        for line_idx, line in enumerate(lines):
            print("Head of line:", len(lines))
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
        plt.title(fn_img)
        plt.show()

if __name__ == '__main__':
    main()
