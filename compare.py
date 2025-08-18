from rembg import remove
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QHBoxLayout, QListWidget, QMessageBox, QScrollArea, QFrame, QDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image
import os



# 去除背景（白边填充）
def rembgImg(image):
    return remove(image)


# 固定尺寸
def resizeImg(image, height=1200):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img


# 边缘检测
def getCanny(image):
    # 高斯模糊
    binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


# 求出面积最大的轮廓
def findMaxContour(image):
    # 寻找边缘
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # 计算面积
    max_area = 0.0
    max_contour = []
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
    return max_contour, max_area


# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


# 适配原四边形点集
def adaPoint(box, pro):
    box_pro = box
    if pro != 1.0:
        box_pro = box / pro
    box_pro = np.trunc(box_pro)
    return box_pro


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
        pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


def extract_document(filepath, outpath=None):
    """提取票据主要区域并保存（如果outpath提供）"""
    image = cv2.imread(filepath)

    # 白边填充背景
    image = rembgImg(image)

    # 尺寸变化
    img = resizeImg(image)
    ratio = 1200 / image.shape[0]  # 放缩尺寸

    # 边缘检测
    binary_img = getCanny(img)

    # 最大轮廓检测
    max_contour, max_area = findMaxContour(binary_img)

    # 轮廓四顶点拟合
    boxes = getBoxPoint(max_contour)

    # 还原原顶点
    boxes = adaPoint(boxes, ratio)

    # 顶点排序
    boxes = orderPoints(boxes)

    # 透视变化
    warped = warpImage(image, boxes)

    if outpath:
        cv2.imwrite(outpath, warped)

    return warped, boxes


def preprocess_for_comparison(image, target_size=(1600, 1200)):
    """预处理图像用于比较"""
    # 统一尺寸
    resized = cv2.resize(image, target_size)

    # 灰度化
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=15
    )
    return binary

def compare_documents(img1_prep, img2_prep, threshold=0.25):
    """
    比较两张提取后的票据图像
    :param img1_prep: 预处理后第一张票据图像
    :param img2_prep: 预处理后第二张票据图像
    :param threshold: 相似度阈值（0-1之间）
    :return: (是否相同, 相似度分数, 可视化图像)
    """

    # 方法1: 结构相似性 (SSIM)
    def calculate_ssim(img1, img2):
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=img2.max() - img2.min())

    ssim_score = calculate_ssim(img1_prep, img2_prep)

    # 方法2: ORB特征匹配
    def orb_feature_matching(img1, img2):
        # 初始化ORB检测器
        orb = cv2.ORB_create(nfeatures=10000)

        # 检测关键点和计算描述符
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 如果没有足够的特征点，返回0
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0, 0, None

        # 创建BF匹配器
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # 应用比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 计算匹配率
        match_rate = len(good_matches) / min(len(des1), len(des2))

        # 绘制匹配结果
        if good_matches:
            matched_img = cv2.drawMatches(
                cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), kp1,
                cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), kp2,
                good_matches[:50], None,  # 最多显示50个匹配点
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        else:
            matched_img = None

        return match_rate, len(good_matches), matched_img

    orb_rate, num_matches, orb_img = orb_feature_matching(img1_prep, img2_prep)

    # 方法3: 直方图比较
    def histogram_comparison(img1, img2):
        # 计算直方图
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

        # 归一化
        cv2.normalize(hist1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # 计算直方图相关性
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    hist_score = histogram_comparison(img1_prep, img2_prep)

    # 综合评分 (加权平均)
    weights = {
        'ssim': 0.5,
        'orb': 0.3,
        'hist': 0.2
    }

    # 如果ORB匹配失败，调整权重
    if orb_rate == 0:
        weights = {
            'ssim': 0.7,
            'orb': 0.0,
            'hist': 0.3
        }
        orb_rate = 0

    similarity_score = (
            weights['ssim'] * ssim_score +
            weights['orb'] * orb_rate +
            weights['hist'] * hist_score
    )

    # 判断结果
    is_same = similarity_score > threshold
    return is_same, similarity_score


def create_pdf_from_images(image_paths, pdf_filename):
    """
    根据图片路径列表创建PDF文件，每页一张图片
    :param image_paths: 图片路径列表（按顺序）
    :param pdf_filename: 输出PDF文件名
    """
    try:
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
        print(f"PDF文件已创建: {pdf_filename}")

    except Exception as e:
        print(f"创建PDF时出错: {e}")


# ========== 工具函数 ==========
def cvimg_to_qpixmap(cv_img):
    """OpenCV 图像转 QPixmap"""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ========== 可点击的 QLabel ==========
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_img = None  # 保存原始 OpenCV 图像

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


# ========== 放大预览窗口 ==========
class ImagePreviewDialog(QDialog):
    def __init__(self, cv_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片预览")
        self.setGeometry(300, 100, 1000, 800)

        vbox = QVBoxLayout()
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)

        # 转换为 QPixmap
        pixmap = cvimg_to_qpixmap(cv_img)
        lbl.setPixmap(pixmap.scaled(1600, 1200, Qt.KeepAspectRatio))

        vbox.addWidget(lbl)
        self.setLayout(vbox)


# ========== 主窗口类 ==========
class ImageCompareApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像相似度比对工具")
        self.setGeometry(100, 50, 1600, 1000)  # 大窗口

        self.images_I = []
        self.images_II = []

        # 上传、删除、比对按钮
        self.btn_upload_I = QPushButton("上传图片组 I")
        self.btn_upload_II = QPushButton("上传图片组 II")
        self.btn_delete = QPushButton("删除选中")
        self.btn_start = QPushButton("开始比对")

        # 图片文件列表
        self.list_I = QListWidget()
        self.list_II = QListWidget()

        # 结果显示区（滚动）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.result_container = QVBoxLayout()
        self.result_widget = QWidget()
        self.result_widget.setLayout(self.result_container)
        self.scroll_area.setWidget(self.result_widget)

        # 主布局
        layout = QVBoxLayout()

        # 顶部两个上传按钮
        upload_btn_layout = QHBoxLayout()
        upload_btn_layout.addWidget(self.btn_upload_I)
        upload_btn_layout.addWidget(self.btn_upload_II)
        layout.addLayout(upload_btn_layout)

        # 文件列表
        lists_layout = QHBoxLayout()
        lists_layout.addWidget(self.list_I)
        lists_layout.addWidget(self.list_II)
        layout.addLayout(lists_layout)

        # 删除按钮
        layout.addWidget(self.btn_delete)

        # 中间的“开始比对”按钮
        layout.addWidget(self.btn_start)

        # 比对结果
        layout.addWidget(QLabel("比对结果"))
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

        # 信号绑定
        self.btn_upload_I.clicked.connect(self.upload_images_I)
        self.btn_upload_II.clicked.connect(self.upload_images_II)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_start.clicked.connect(self.start_compare)

    # 上传图片组 I（追加）
    def upload_images_I(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片 I", "", "Images (*.png *.jpg *.jpeg)")
        if files:
            self.images_I.extend(files)
            self.list_I.addItems(files)

    # 上传图片组 II（追加）
    def upload_images_II(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片 II", "", "Images (*.png *.jpg *.jpeg)")
        if files:
            self.images_II.extend(files)
            self.list_II.addItems(files)

    # 删除选中项（两个列表都检查）
    def delete_selected(self):
        selected_I = self.list_I.selectedItems()
        for item in selected_I:
            path = item.text()
            if path in self.images_I:
                self.images_I.remove(path)
            self.list_I.takeItem(self.list_I.row(item))

        selected_II = self.list_II.selectedItems()
        for item in selected_II:
            path = item.text()
            if path in self.images_II:
                self.images_II.remove(path)
            self.list_II.takeItem(self.list_II.row(item))

    # 清空历史结果
    def clear_results(self):
        for i in reversed(range(self.result_container.count())):
            widget = self.result_container.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    # 开始比对
    def start_compare(self):
        if not self.images_I or not self.images_II:
            QMessageBox.warning(self, "警告", "请先上传两组图片")
            return
        if len(self.images_I) != len(self.images_II):
            QMessageBox.warning(self, "警告", "两组图片数量必须相同")
            return

        self.clear_results()

        #存储预处理结果
        preprocess_for_imgI = {}
        preprocess_for_imgII = {}

        # 存储匹配结果，用于PDF生成
        matched_pairs = []

        for imgI_path in self.images_I:
            imgI, _ = extract_document(imgI_path)
            preprocess_for_imgI[imgI_path] = preprocess_for_comparison(imgI)

        for imgII_path in self.images_II:
            imgII = cv2.imread(imgII_path)
            preprocess_for_imgII[imgII_path] = preprocess_for_comparison(imgII)


        for i, imgI_path in enumerate(self.images_I):
            imgI = preprocess_for_imgI[imgI_path]
            best_score = -1
            best_match_path = None
            best_imgII = None

            for imgII_path, imgII in preprocess_for_imgII.items():
                _, score = compare_documents(imgI, imgII)
                if score > best_score:
                    best_score = score
                    best_match_path = imgII_path
                    best_imgII = imgII

            # 保存匹配对
            matched_pairs.append((imgI_path, best_match_path, best_score))

            # GUI 显示结果
            frame = QFrame()
            frame_layout = QVBoxLayout()

            row = QHBoxLayout()

            lblI = ClickableLabel()
            lblII = ClickableLabel()

            # 保存原始图像用于放大
            lblI.cv_img = imgI
            lblII.cv_img = best_imgII

            # 缩略图
            lblI.setPixmap(cvimg_to_qpixmap(imgI).scaled(600, 450, Qt.KeepAspectRatio))
            lblII.setPixmap(cvimg_to_qpixmap(best_imgII).scaled(600, 450, Qt.KeepAspectRatio))

            # 点击事件：弹出大图
            lblI.clicked.connect(lambda img=lblI.cv_img: self.show_preview(img))
            lblII.clicked.connect(lambda img=lblII.cv_img: self.show_preview(img))

            row.addWidget(lblI)
            row.addWidget(lblII)
            frame_layout.addLayout(row)

            score_label = QLabel(
                f"图片 I {i+1}\n最佳匹配: {best_match_path}\n相似度 = {best_score:.4f}"
            )
            score_label.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(score_label)

            frame.setLayout(frame_layout)
            self.result_container.addWidget(frame)

        # 生成PDF文件
        try:

            # 分别提取两组图片路径
            pdf_I_paths = [pair[0] for pair in matched_pairs]
            pdf_II_paths = [pair[1] for pair in matched_pairs]

            # 生成PDF文件
            create_pdf_from_images(pdf_I_paths, "图片组I.pdf")
            create_pdf_from_images(pdf_II_paths, "图片组II.pdf")

            QMessageBox.information(self, "成功",
                                    "比对完成！\n已生成PDF文件：\n- 图片组I_按相似度排序.pdf\n- 图片组II_按相似度排序.pdf")

        except Exception as e:
            QMessageBox.warning(self, "PDF生成失败", f"生成PDF时出错：{str(e)}")

    def show_preview(self, cv_img):
        """弹出放大预览窗口"""
        dlg = ImagePreviewDialog(cv_img, self)
        dlg.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageCompareApp()
    window.show()
    sys.exit(app.exec_())
