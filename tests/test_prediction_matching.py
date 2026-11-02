from pathlib import Path
import numpy as np
from course.lab_utils import match_predictions, label_path_for_image

class Array:
    def __init__(self, x):self.x=np.asarray(x)
    def cpu(self):return self
    def numpy(self):return self.x

class Boxes:
    xyxy=Array([[40,40,60,60],[70,70,90,90],[10,10,20,20]])
    cls=Array([0,1,0])
    conf=Array([.92,.72,.49])

class Result:
    orig_shape=(100,100)
    boxes=Boxes()


def test_prediction_matching_fixed_threshold(tmp_path):
    labels=tmp_path/'img.txt'
    labels.write_text('0 0.5 0.5 0.2 0.2\n0 0.8 0.8 0.2 0.2\n')
    r=match_predictions(Result(),labels,0.50)
    assert (r['TP'],r['FP'],r['FN'])==(1,2,1)
    assert len(r['errors'])==3


def test_label_path_nested(tmp_path):
    p=tmp_path/'images/val/subfolder/x.jpg'
    assert label_path_for_image(p)==tmp_path/'labels/val/subfolder/x.txt'
