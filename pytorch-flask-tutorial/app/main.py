from flask import Flask, request, jsonify

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})
    # 1 load image
    # 2 image -> tensor
    # 3 prediction
    # 4 return json
    return jsonify({'result': 1})

@app.route('/', methods=['GET'])
def upload_form():
    return '''
        <html>
            <body>
                <h1>Upload an Image for Prediction</h1>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*">
                    <input type="submit" value="Upload">
                </form>
            </body>
        </html>
    '''
