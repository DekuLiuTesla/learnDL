import glob
import os
import numpy as np
import json

if __name__ == '__main__':
    path = 'D:\Projects\SDPG\\apollocar3d_models_json'
    models = glob.glob(os.path.join(path, '*.json'))
    print(f'Number of Models: {len(models)}')

    bbox_dim_list = []
    max_num_points = 0
    for i, model_path in enumerate(models):
        model = json.load(open(model_path))
        vertices = np.array(model['vertices'])[:, (2, 0, 1)]
        max_vertices = np.max(vertices, axis=0)
        min_vertices = np.min(vertices, axis=0)
        bbox_dim = max_vertices-min_vertices
        bbox_dim_list.append(bbox_dim)
        max_num_points = max_num_points if max_num_points > vertices.shape[0] else vertices.shape[0]
        print(model['car_type'], ': ', bbox_dim)

    bbox_dim_array = np.array(bbox_dim_list)
    max_dim = np.max(bbox_dim_array, axis=0)
    min_dim = np.min(bbox_dim_array, axis=0)
    print('\n')
    print(f'Max dim of models: {max_dim}')
    print(f'Min dim of models: {min_dim}')
    print(f'Max number of points: {max_num_points}')
    print('Done')
