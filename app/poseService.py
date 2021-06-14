import torch
from manual_poser import ManualPoserApp

from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from tha.combiner import CombinerSpec
from tha.face_morpher import FaceMorpherSpec
from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def pose():
    cuda = torch.device('cuda')
    poser = MorphRotateCombinePoser256Param6(
        morph_module_spec=FaceMorpherSpec(),
        morph_module_file_name="data/face_morpher.pt",
        rotate_module_spec=TwoAlgoFaceRotatorSpec(),
        rotate_module_file_name="data/two_algo_face_rotator.pt",
        combine_module_spec=CombinerSpec(),
        combine_module_file_name="data/combiner.pt",
        device=cuda)

    if request.method == 'POST':
        start_x = request.form.get('start_x')
        start_y = request.form.get('start_y')
        start_z = request.form.get('start_z')
        start_left_eye = request.form.get('start_left_eye')
        start_right_eye = request.form.get('start_right_eye')
        start_mouth = request.form.get('start_mouth')

        end_x = request.form.get('end_x')
        end_y = request.form.get('end_y')
        end_z = request.form.get('end_z')
        end_left_eye = request.form.get('end_left_eye')
        end_right_eye = request.form.get('end_right_eye')
        end_mouth = request.form.get('end_mouth')

        total_frame_number = request.form.get('total_frame_number')
        startPose = [float(start_x), float(start_y), float(start_z), float(start_left_eye), float(start_right_eye),
                     float(start_mouth)]
        endPose = [float(end_x), float(end_y), float(end_z), float(end_left_eye), float(end_right_eye),
                   float(end_mouth)]

    if request.method == 'GET':
        startPose = [-1.0, -1.0, -1.0, 0, 0, 0]
        endPose = [1.0, 1.0, 1.0, 1, 1, 1]
        total_frame_number = 24
    ManualPoserApp(poser=poser, torch_device=cuda).main_loop(startPose, endPose, int(total_frame_number))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug = True)
