import fitz
import os
from win32com import client

def word_to_pdf(filepath):
    targetpath = os.path.splitext(filepath)[0] + '.pdf'
    word = client.Dispatch('Word.Application')
    doc = word.Documents.Open(filepath)
    doc.SaveAs(f"{targetpath}", FileFormat=17)
    doc.Close()
    word.Quit()
    return targetpath

def pdf_to_jpg(filepath):
    filename = os.path.splitext(os.path.basename(filepath))

    # convert word to pdf
    if filename[1] != 'pdf':
        filepath = word_to_pdf(filepath)

    doc = fitz.open(filepath)
    for pg_num in range(len(doc)):
        page = doc.load_page(pg_num)
        pix = page.get_pixmap(dpi=300)
        pix.save(f'{filename[0]}_{pg_num + 1}.jpg')
    doc.close()

    # convert word to pdf
    if filename[1] != 'pdf':
        os.remove(filepath)

if __name__ == '__main__':
    pdf_to_jpg('E:\Pycharm2023\pycharm\ImageRecognition\wordex.docx')