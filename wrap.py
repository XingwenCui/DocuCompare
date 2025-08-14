from paddleocr import TextImageUnwarping

model = TextImageUnwarping(model_name="UVDoc")
output = model.predict("rotate2.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="test5.png")