import glob
import os
import numpy as np
import json

if __name__ == '__main__':
    path = 'D:\Projects\SDPG\\apollocar3d_models_json'
    models = glob.glob(os.path.join(path, '*.json'))
    for i, model_path in enumerate(models):
        model = json.load(open(model_path))
        vertices = np.array(model['vertices'])[:, (2, 0, 1)]
        max_vertices = np.max(vertices, axis=0)
        min_vertices = np.min(vertices, axis=0)
        bbox_dim = max_vertices-min_vertices
        print(model['car_type'], ': ', bbox_dim)
    print('Done')
