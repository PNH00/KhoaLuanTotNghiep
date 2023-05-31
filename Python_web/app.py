from flask import Flask, render_template, request
from PIL import Image,ImageEnhance
import io as ib
import numpy as np
import cv2
from SIRR import SIRR, SIRRImprove
import base64

#Tăng độ sáng cho ảnh đã tách phản chiếu 
def brightness_filter(image):
    img = Image.fromarray(np.uint8(image*255))
    curr_bri = ImageEnhance.Brightness(img)
    new_bri = 1.5
    img_brightened = curr_bri.enhance(new_bri)
    return img_brightened

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		# Lấy ảnh từ form
		image_file = request.files['image']
		# Đọc ảnh từ file
		img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
		img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# Áp dụng SIRR
		T,R = SIRR(img_input,10)
		#Tăng độ sáng 
		img_output = brightness_filter(T)
		#Lưu ảnh vào bộ nhớ đệm
		buffered = ib.BytesIO()
		img_output.save(buffered, format='JPEG')
		buffered.seek(0)
		#show ảnh đã sử lý
		img_SIR = base64.b64encode(buffered.getvalue()).decode('utf-8')
		#Hiển thị ảnh gốc
		img_ORG = cv2.imencode('.jpg', img)[1].tobytes()
		img_ORG_base64 = base64.b64encode(img_ORG).decode('utf-8')
		return render_template('index_Result.html', img_SIRR=img_SIR,img_ORG = img_ORG_base64)
	return render_template('index.html')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port='80')