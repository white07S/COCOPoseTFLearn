import numpy as np

def config_reader():
    config = {
        'param': {
            'use_gpu': 1,
            'GPUdeviceNumber': 0,
            'modelID': 1,
            'octave': 3,
            'starting_range': 0.8,
            'ending_range': 2,
            'scale_search': [0.5, 1, 1.5, 2],
            'thre1': 0.1,
            'thre2': 0.05,
            'thre3': 0.5,
            'min_num': 4,
            'mid_num': 10,
            'crop_ratio': 2.5,
            'bbox_ratio': 0.25
        },
        'models': {
            '1': {
                'boxsize': 368,
                'padValue': 128,
                'np': 12,
                'stride': 8,
                'part_str': ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Leye", "Reye", "Lear", "Rear", "pt19"]
            }
        }
    }

    param = config['param']
    model_id = param['modelID']
    model = config['models'][str(model_id)]

    param['octave'] = int(param['octave'])
    param['use_gpu'] = int(param['use_gpu'])
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = list(map(float, param['scale_search']))
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model

if __name__ == "__main__":
    param, model = config_reader()
