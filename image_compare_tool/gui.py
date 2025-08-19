import os
import cv2
import shutil
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QHBoxLayout, QListWidget, QMessageBox, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt

from image_processing import extract_document, preprocess_for_comparison, compare_documents, imread_unicode
from pdf_utils import pdf_to_images, create_pdf_from_images
from utils import cvimg_to_qpixmap
from widgets import ClickableLabel, ImagePreviewDialog


class ImageCompareApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像相似度比对工具")
        self.setGeometry(100, 50, 1600, 1000)

        self.images_I = []
        self.images_II = []
        self.temp_dirs = []     # PDF 转换的临时目录

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

        # 布局
        layout = QVBoxLayout()

        upload_btn_layout = QHBoxLayout()
        upload_btn_layout.addWidget(self.btn_upload_I)
        upload_btn_layout.addWidget(self.btn_upload_II)
        layout.addLayout(upload_btn_layout)

        lists_layout = QHBoxLayout()
        lists_layout.addWidget(self.list_I)
        lists_layout.addWidget(self.list_II)
        layout.addLayout(lists_layout)

        layout.addWidget(self.btn_delete)
        layout.addWidget(self.btn_start)
        layout.addWidget(QLabel("比对结果"))
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

        # 信号绑定
        self.btn_upload_I.clicked.connect(self.upload_images_I)
        self.btn_upload_II.clicked.connect(self.upload_images_II)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_start.clicked.connect(self.start_compare)

    # 上传图片组 I
    def upload_images_I(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片或PDF I", "", "Images/PDF (*.png *.jpg *.jpeg *.pdf)"
        )
        if files:
            for f in files:
                if f.lower().endswith(".pdf"):
                    img_paths = pdf_to_images(f, self.temp_dirs)
                    self.images_I.extend(img_paths)
                    self.list_I.addItems(img_paths)
                else:
                    self.images_I.append(f)
                    self.list_I.addItem(f)

    # 上传图片组 II
    def upload_images_II(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片或PDF II", "", "Images/PDF (*.png *.jpg *.jpeg *.pdf)"
        )
        if files:
            for f in files:
                if f.lower().endswith(".pdf"):
                    img_paths = pdf_to_images(f, self.temp_dirs)
                    self.images_II.extend(img_paths)
                    self.list_II.addItems(img_paths)
                else:
                    self.images_II.append(f)
                    self.list_II.addItem(f)

    # 删除选中项
    def delete_selected(self):
        for item in self.list_I.selectedItems():
            path = item.text()
            if path in self.images_I:
                self.images_I.remove(path)
            self.list_I.takeItem(self.list_I.row(item))

        for item in self.list_II.selectedItems():
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
        preprocess_for_imgI = {}
        preprocess_for_imgII = {}
        matched_pairs = []

        # 预处理
        for imgI_path in self.images_I:
            imgI, _ = extract_document(imgI_path)
            preprocess_for_imgI[imgI_path] = preprocess_for_comparison(imgI)

        for imgII_path in self.images_II:
            imgII = imread_unicode(imgII_path)
            preprocess_for_imgII[imgII_path] = preprocess_for_comparison(imgII)

        # 比对
        for i, imgI_path in enumerate(self.images_I):
            imgI = preprocess_for_imgI[imgI_path]
            best_score, best_match_path, best_imgII = -1, None, None

            for imgII_path, imgII in preprocess_for_imgII.items():
                _, score = compare_documents(imgI, imgII)
                if score > best_score:
                    best_score, best_match_path, best_imgII = score, imgII_path, imgII

            matched_pairs.append((imgI_path, best_match_path, best_score))

            # GUI 显示结果
            frame, frame_layout = QFrame(), QVBoxLayout()
            row = QHBoxLayout()

            lblI, lblII = ClickableLabel(), ClickableLabel()
            lblI.cv_img, lblII.cv_img = imgI, best_imgII
            lblI.setPixmap(cvimg_to_qpixmap(imgI).scaled(600, 450, Qt.KeepAspectRatio))
            lblII.setPixmap(cvimg_to_qpixmap(best_imgII).scaled(600, 450, Qt.KeepAspectRatio))

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

        # 生成 PDF
        try:
            pdf_I_paths = [pair[0] for pair in matched_pairs]
            pdf_II_paths = [pair[1] for pair in matched_pairs]
            create_pdf_from_images(pdf_I_paths, "图片组I.pdf")
            create_pdf_from_images(pdf_II_paths, "图片组II.pdf")

            QMessageBox.information(
                self, "成功",
                "比对完成！\n已生成PDF文件：\n- 图片组I.pdf\n- 图片组II.pdf"
            )
        except Exception as e:
            QMessageBox.warning(self, "PDF生成失败", f"生成PDF时出错：{str(e)}")

    # 图片预览
    def show_preview(self, cv_img):
        dlg = ImagePreviewDialog(cv_img, self)
        dlg.exec_()

    # 关闭时清理临时文件
    def closeEvent(self, event):
        for dir in self.temp_dirs:
            try:
                shutil.rmtree(dir)
            except Exception as e:
                print(f"删除临时目录失败: {dir}, 错误: {e}")
        event.accept()
