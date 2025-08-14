import json
import numpy as np
from difflib import SequenceMatcher, ndiff


def merge_ocr_lines(ocr_json, score_thresh=0.8, y_thresh=20):
    """
    合并 PaddleOCR JSON 结果中的碎片文字
    """
    polys = ocr_json["dt_polys"]
    texts = ocr_json["rec_texts"]
    scores = ocr_json["rec_scores"]

    # 计算每个检测框的中心点
    centers = [np.mean(p, axis=0) for p in polys]

    # 过滤低置信度文字
    items = []
    for i in range(len(texts)):
        if scores[i] >= score_thresh:
            items.append((centers[i][0], centers[i][1], texts[i]))

    # 按 y 坐标排序
    items.sort(key=lambda x: x[1])

    lines = []
    current_line = []
    last_y = None

    for x, y, text in items:
        if last_y is None or abs(y - last_y) < y_thresh:  # 同一行
            current_line.append((x, text))
        else:  # 新的一行
            current_line.sort(key=lambda i: i[0])
            lines.append("".join(t for _, t in current_line))
            current_line = [(x, text)]
        last_y = y

    if current_line:
        current_line.sort(key=lambda i: i[0])
        lines.append("".join(t for _, t in current_line))

    return lines

def similarity(s1, s2):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, s1, s2).ratio()

def print_diff_lines(lines1, lines2):
    """
    按行对比两份文本，显示差异
    """
    max_len = max(len(lines1), len(lines2))
    for i in range(max_len):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""

        if line1 != line2:
            print(f"\n第 {i+1} 行不同：")
            for diff in ndiff([line1], [line2]):
                print(diff)

def compare_ocr_json(file1, file2, sim_thresh=0.95):
    """
    比较两个 PaddleOCR JSON 文件的文本内容（带多行差异标注）
    """
    with open(file1, "r", encoding="utf-8") as f:
        ocr1 = json.load(f)
    with open(file2, "r", encoding="utf-8") as f:
        ocr2 = json.load(f)

    # 合并行文字
    merged1 = merge_ocr_lines(ocr1)
    merged2 = merge_ocr_lines(ocr2)

    # 拼接成一个整体文本
    text1 = "\n".join(merged1)
    text2 = "\n".join(merged2)

    # 计算相似度
    score = similarity(text1, text2)

    print("=== 票据1文本 ===")
    print(text1)
    print("\n=== 票据2文本 ===")
    print(text2)

    print(f"\n相似度: {score:.2%}")
    if score >= sim_thresh:
        print("✅ 两张票据内容一致")
    else:
        print("❌ 两张票据内容不一致")
        print_diff_lines(merged1, merged2)

compare_ocr_json('test2_res.json','test4_res.json')
