from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import sys
from src.utils.logger import logger
from src.utils.exceptions import CustomException
from src.prediction.predict import ImagePredictor

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            raise CustomException('No file part', sys.exc_info())

        file = request.files['file']

        if file.filename == '':
            raise CustomException('No selected file', sys.exc_info())

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = f"data/raw/{filename}"
            file.save(file_path)

            image_predictor = ImagePredictor()
            result_df = image_predictor.predict_image(file_path)

            # Process result_df as needed
            return render_template('index.html', result=result_df.to_html())
    except CustomException as ce:
        return render_template('index.html', error=str(ce))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return render_template('index.html', error="Unexpected error. Please check logs.")

    return render_template('index.html', error="Invalid file format. Allowed formats: png, jpg, jpeg, gif.")

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8000)
