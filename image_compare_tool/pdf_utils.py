import os
import fitz
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image
from win32com import client

def word_to_pdf(filepath, targetdir):
    targetpath = os.path.join(targetdir, os.path.splitext(os.path.split(filepath)[-1])[0] + '.pdf')
    print(targetpath)
    word = client.Dispatch('Word.Application')
    doc = word.Documents.Open(filepath)
    doc.SaveAs(f"{targetpath}", FileFormat=17)
    doc.Close()
    word.Quit()
    return targetpath


def pdf_to_images(pdf_path, temp_dirs):
    filename = os.path.splitext(os.path.basename(pdf_path))

    temp_dir = tempfile.mkdtemp(dir=os.path.dirname(pdf_path))
    temp_dirs.append(temp_dir)
    print(temp_dir)

    if filename[1] != 'pdf':
        pdf_path = word_to_pdf(pdf_path, temp_dir)

    doc = fitz.open(pdf_path)
    img_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(temp_dir, f"{os.path.basename(pdf_path)}_page{page_num + 1}.png")
        pix.save(img_path)
        img_paths.append(img_path)

    doc.close()
    return img_paths


def create_pdf_from_images(image_paths, pdf_filename):
    """
    根据图片路径列表创建PDF文件，每页一张图片
    :param image_paths: 图片路径列表（按顺序）
    :param pdf_filename: 输出PDF文件名
    """
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4

    for i, img_path in enumerate(image_paths):
        if i > 0:  # 从第二张图片开始，先添加新页面
            c.showPage()
        try:
            # 使用PIL打开图片
            pil_img = Image.open(img_path)

            # 获取图片尺寸
            img_width, img_height = pil_img.size

            # 计算缩放比例，保持宽高比
            scale_w = page_width / img_width
            scale_h = page_height / img_height
            scale = min(scale_w, scale_h) * 0.9  # 留一些边距

            # 计算居中位置
            new_width = img_width * scale
            new_height = img_height * scale
            x = (page_width - new_width) / 2
            y = (page_height - new_height) / 2

            # 将图片绘制到PDF
            c.drawImage(img_path, x, y, width=new_width, height=new_height)

        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            # 在页面上添加错误信息
            c.drawString(50, page_height - 50, f"无法加载图片: {os.path.basename(img_path)}")

    c.save()