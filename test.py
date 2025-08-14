from paddleocr import PaddleOCR

# 文本检测+文本识别
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

result = ocr.predict("test4.png")
for res in result:
    res.save_to_json("./")
    res.save_to_img("./")
