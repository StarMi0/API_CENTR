import tensorflow as tf
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    # tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )

import time, os, werkzeug, zipfile, cv2, shutil
from flask import Flask, render_template, make_response, request, Blueprint, send_file
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.utils import secure_filename
from functions import get_img_for_predict, predict_img, save_gen_img

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# допустимые разсширения имен файлов с изображениями
ALLOWED_EXTENSIONS = {'zip', 'rar', '7z', 'tar', 'gz'}

# имя каталога для загрузки изображений
UPLOAD_FOLDER = 'inputs/'
RESULT_FOLDER = 'result/'

# g_model and weights downloading
model_name1 = 'models/M_6.h5'


def allowed_file(file_name):
    """
    Функция проверки расширения файла
    """
    return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_affix(filename, affix='_CENTR'):
    name, extension = os.path.splitext(filename)
    return name + affix + '.png'


def zip_folder(name):
    print(name)
    zip_name = name + '.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(name):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))

    zip_ref.close()


# создаем WSGI приложение
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# создаем API-сервер
api = Api(app)

# создаем парсер API-запросов
parser = reqparse.RequestParser()
parser.add_argument('image_file', type=werkzeug.datastructures.FileStorage,
                    help='Binary Image in png format (zip, rar, 7z, tar, gz)', location='files', required=True)
parser.add_argument('scale', type=werkzeug.datastructures.Accept, required=True)


@api.route('/images', methods=['GET', 'POST'])
@api.produces(['/application'])
class Images(Resource):
    # если POST-запрос
    @api.expect(parser)
    def post(self):
        try:
            start_time = time.time()

            if 'image_file' not in request.files:
                raise ValueError('No input file')

            f = request.files['image_file']
            req_mirr = request.form.get('mirr');

            if f.filename == '':
                raise ValueError('Empty file name')

            if not allowed_file(f.filename):
                raise ValueError('Unsupported file type')

            image_file_name = secure_filename(f.filename)
            image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file_name)
            f.save(image_file_path)

            unzipped_folder = os.path.join(app.config['UPLOAD_FOLDER'],
                                           image_file_name.split('.')[0] + time.time().__str__())
            result_folder = os.path.join(app.config['RESULT_FOLDER'], image_file_name.split('.')[0])

            if not os.path.exists(result_folder):
                os.mkdir(result_folder)

            with zipfile.ZipFile(image_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_folder)

            time.sleep(3)

            # Pass the unzipped folder to get_img_for_predict
            img_for_pred = get_img_for_predict(unzipped_folder)
            print('image is collected')

            # Make predictions
            if req_mirr == 'mirr1':
                imgs = predict_img(img_for_pred, model_name1)

            save_gen_img(imgs, result_folder)

            for filename in os.listdir(result_folder):
                if filename.endswith('.png'):
                    img = cv2.imread(os.path.join(result_folder, filename))
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(result_folder, filename), gray_image)

            zip_folder(result_folder)

            os.remove(image_file_path)
            shutil.rmtree(unzipped_folder)
            shutil.rmtree(result_folder)

            zip_file = result_folder + '.zip';

            if req_mirr == 'mirr2':
                zip_file = 'left_masks/left_masks.zip'

            response = send_file(zip_file, download_name=os.path.basename(zip_file), as_attachment=True,
                                 mimetype='application/zip')
            os.remove(zip_file)
            return response

        except ValueError as err:
            dict_response = {
                'error': err.args[0],
                'filename': f.filename,
                'time': (time.time() - start_time)
            }
            return dict_response

        except:
            dict_response = {
                'error': 'Unknown error',
                'time': (time.time() - start_time)
            }
            return dict_response


# запускаем сервер на порту 8008 (или на любом другом свободном порту)
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)

