import os
import shutil

if __name__ == "__main__":
    # index
    index_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/index"
    indexes = [t[:-4] for t in os.listdir(index_path) if t.endswith(".png")]

    # select
    src_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/calibdata1208_yuv444"
    dist_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/calibration1208_yuv444_sub"
    os.makedirs(dist_path, exist_ok=True)
    os.makedirs(os.path.join(dist_path, "infra1"), exist_ok=True)
    os.makedirs(os.path.join(dist_path, "infra2"), exist_ok=True)
    for index in indexes:
        shutil.copy(os.path.join(src_path, "infra1", index + ".npy"), os.path.join(dist_path, "infra1", index + ".npy"))
        shutil.copy(os.path.join(src_path, "infra2", index + ".npy"), os.path.join(dist_path, "infra2", index + ".npy"))
