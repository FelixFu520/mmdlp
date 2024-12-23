import json
import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2


def find_jpg_files(root_dir):
    jpg_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                jpg_files.append(file_path)
    return jpg_files

def search_jpg(all_images, image_name):
    for image in all_images:
        if image_name in image:
            return image
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overkill background')
    parser.add_argument('--preds_path', type=str, default='/home/users/public/fa.fu/overkill_test/wushibie_results_20241215', help='path to the prediction result')
    parser.add_argument('--images_path', type=str, default='/home/users/public/fa.fu/overkill_test/images', help='path to the images')
    parser.add_argument('--score_step', type=float, default=0.05, help='score step')
    parser.add_argument('--show_dir', type=str, default='/home/users/public/fa.fu/overkill_test/show_dir', help='show dir')
    parser.add_argument('--roi', type=bool, default=True, help='roi')
    parser.add_argument('--exclude_class', type=str, default="skein", help='gpu')
    args = parser.parse_args()

    preds_path = args.preds_path
    images_path = args.images_path
    score_step = args.score_step
    show_dir = args.show_dir
    exclude_class = [e for e in args.exclude_class.split(";")]
    
    # 获得所有预测的文件夹和每张图的预测结果
    all_dirs = [os.path.join(preds_path, d) for d in os.listdir(preds_path)]
    print(f"find {len(all_dirs)} dirs")
    all_preds = []
    for d in all_dirs:
        preds = [os.path.join(d, f) for f in os.listdir(d)]
        print(f"find {len(preds)} preds in {d}")
        all_preds.extend(preds)
    
    print(f"find {len(all_preds)} preds in total")

    # 获得所有图片路径
    all_images = find_jpg_files(images_path)
    print(f"find {len(all_images)} images in total")

    # 统计误检
    all_scores = {} # key:score, value: overkill_num

    for score in np.around(np.arange(0, 1.05, 0.05), 2):
        overkill_num = 0    # 误检数量
        os.makedirs(os.path.join(show_dir, str(score)), exist_ok=True)
        for pred_json_p in tqdm(all_preds, desc=f"processing score {score}"):
            with open(pred_json_p, 'r') as f:
                pred_json = json.load(f)
                if len(pred_json['detections'])  != 0:
                    # 获得图片路径
                    image_name = os.path.basename(pred_json_p).split('.mix.json')[0] + ".jpg"
                    image_p = search_jpg(all_images, image_name)
                    image = None if image_p is None else cv2.imread(image_p)
                    height, width, _ = image.shape if image is not None else (None, None, None)
                    assert height is not None and width is not None, f"image shape is {height} {width}"

                    for detection in pred_json['detections']:   # detection['boundingBox'] = [xmin, ymin, width, height]
                        # 筛选1. ROI
                        if args.roi:
                            if (int(detection['boundingBox'][1])+int(detection['boundingBox'][3])) <= 521:  # 去掉上方520像素， ymax<520则忽略
                                continue
                            
                            # 更新ymin和height
                            detection['boundingBox'][1] = max(0, detection['boundingBox'][1]-520)   # 确保ymin>=0
                            detection['boundingBox'][3] = (height - detection['boundingBox'][1]) if (detection['boundingBox'][1] + detection['boundingBox'][3]) > height else detection['boundingBox'][3]
                        # 筛选2. 排除指定类别
                        if detection['labelName'] in exclude_class:
                            continue

                        # 筛选3. 只保留score大于等于阈值的
                        if detection['score'] >= score:
                            overkill_num += 1
                            # 可视化
                            if image is not None:
                                if args.roi:
                                    cv2.line(image, (0, 520), (width, 520), (255, 0, 0), 2)
                                    cv2.rectangle(
                                        image, 
                                        (int(detection['boundingBox'][0]), int(detection['boundingBox'][1]+ 520) ),
                                        (int(detection['boundingBox'][0])+int(detection['boundingBox'][2]), int(detection['boundingBox'][1]+520)+int(detection['boundingBox'][3])),
                                        (0, 0, 255),
                                        2)
                                    cv2.putText(
                                        image, 
                                        str(np.round(detection['score'], 4)),
                                        (int(detection['boundingBox'][0]), int(detection['boundingBox'][1]+ 520)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 
                                        1)
                                    cv2.putText(
                                        image, 
                                        str(detection['labelName']),
                                        (int(detection['boundingBox'][0]), int(detection['boundingBox'][1]+ 520) + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 
                                        1)
                                else:
                                    cv2.rectangle(
                                        image, 
                                        (int(detection['boundingBox'][0]), int(detection['boundingBox'][1])),
                                        (int(detection['boundingBox'][0])+int(detection['boundingBox'][2]), int(detection['boundingBox'][1])+int(detection['boundingBox'][3])),
                                        (0, 0, 255),
                                        2)
                                    cv2.putText(
                                        image, 
                                        str(np.round(detection['score'], 4)),
                                        (int(detection['boundingBox'][0]), int(detection['boundingBox'][1])), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 
                                        1)
                                    cv2.putText(
                                        image, 
                                        str(detection['labelName']),
                                        (int(detection['boundingBox'][0]), int(detection['boundingBox'][1]) + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 
                                        1)
                                cv2.imwrite(os.path.join(show_dir, str(score), image_name), image)
                            else:
                                print(f"image not found, {pred_json_p}")
        all_scores[score] = overkill_num
    
    print(all_scores)
    print("------------")
    for k,v in all_scores.items():
        print(f"{k} {v}")