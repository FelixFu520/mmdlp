# DOSOD

```
    parser.add_argument("--data_dir", type=str, 
                        default= "/home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1", 
                        help="The directory of calibration images")
    parser.add_argument("--save_dir", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1_rgb",
                        help="The directory to save calibration data")
    parser.add_argument("--height", type=int,
                        default=672,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=896,
                        help="width")
    args = parser.parse_args()

PYTHONPATH=/home/fa.fu/work/mmdlp/ python3 /home/fa.fu/work/mmdlp/tools/open_explorer/dosod_debug/infer_original_onnx.py \
    --onnx_path  /home/fa.fu/work/work_dirs/dosod_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896.onnx \
    --image_path /home/fa.fu/work/work_dirs/dosod_debug/demo_images/0892.jpg \
    --height 672 \
    --width 896 \
    --result_dir /home/fa.fu/work/work_dirs/dosod_debug/result/

hb_mapper makertbin -c /home/fa.fu/work/mmdlp/tools/open_explorer/dosod_debug/20241124/20241124_v2.yaml--model-type onnx

PYTHONPATH=/home/fa.fu/work/mmdlp/ python3 /home/fa.fu/work/mmdlp/tools/open_explorer/dosod_debug/infer_quantized_onnx.py \
    --onnx_path  /home/fa.fu/work/work_dirs/dosod_debug/20241124/output_v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896_quantized_model.onnx \
    --image_path /home/fa.fu/work/work_dirs/dosod_debug/demo_images/0892.jpg \
    --height 672 \
    --width 896 \
     --result_dir /home/fa.fu/work/work_dirs/dosod_debug/result/
```