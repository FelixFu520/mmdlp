import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, 
                        default="person,head,hand,arm,body,leg,foot,whiteboard,keyboard,mouse,laptop,marker pen,cup,bottle,eraser,microphone,mobile phone",
                        help="Comma-separated text to convert to JSON")
    parser.add_argument("--output_file", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/yoloworld-x5/result/custom_texts.json",)
    args = parser.parse_args()

    # TODO: 自定义类别根据项目需要进行修改
    text = args.text

    # 将文本按逗号分割并去除多余的空格
    items = [item.strip() for item in text.split(",")]

    # 将每个项目转换为单独的列表
    nested_items = [[item] for item in items]

    print("len nested_items:", len(nested_items), nested_items)

    # 指定输出的 JSON 文件名
    output_file = args.output_file

    # 将嵌套列表保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as file:
        # indent=4
        json.dump(nested_items, file, ensure_ascii=False)

    print(f"已成功将数据保存到 {output_file}")