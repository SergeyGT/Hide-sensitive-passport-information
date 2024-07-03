from flask import Flask, request, render_template, url_for, redirect, send_from_directory
import os
import subprocess
from werkzeug.utils import secure_filename
import time


# Сайт
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Папка для сохранения загруженных файлов
app.config['PROCESSED_FOLDER'] = 'processed'  # Папка для сохранения обработанных файлов

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'mp4'}

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' not in request.files:
        return 'Файл не выбран'
    photo = request.files['photo']
    if photo.filename == '':
        return 'Файл не выбран'
    if photo:
        filename = secure_filename(photo.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)

        # Вызов main функции
        result = subprocess.run(['python', 'main.py', '--input', filepath],
                                capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return "Произошла ошибка при обработке изображения."

        # Обработанное изображение сохраняется как 'output.jpg'
        processed_filename = 'output.jpg'

        return render_template('index.html', filename=processed_filename)
    return 'Неверный формат файла'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
        return send_from_directory('', filename)


if __name__ == '__main__':
    app.run(debug=True)
