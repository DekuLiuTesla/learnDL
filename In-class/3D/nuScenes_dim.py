import os
import glob
import numpy as np

if __name__ == '__main__':
    path = 'D:\Projects\SDPG\\nuscene_models'
    models = glob.glob(os.path.join(path, '*.obj'))
    print(f'Number of Models: {len(models)}')

    bbox_dim_list = []
    max_num_points = 0
    for i, model_path in enumerate(models):
        with open(model_path) as file:
            points = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "vt":
                    break
        points = np.array(points)
        max_vertices = np.max(points, axis=0)
        min_vertices = np.min(points, axis=0)
        bbox_dim = max_vertices-min_vertices
        bbox_dim_list.append(bbox_dim)
        max_num_points = max_num_points if max_num_points > points.shape[0] else points.shape[0]
        car_name = os.path.splitext(os.path.basename(model_path))[0]
        print(car_name, ': ', bbox_dim)

    bbox_dim_array = np.array(bbox_dim_list)
    max_dim = np.max(bbox_dim_array, axis=0)
    min_dim = np.min(bbox_dim_array, axis=0)
    print('\n')
    print(f'Max dim of models: {max_dim}')
    print(f'Min dim of models: {min_dim}')
    print(f'Max number of points: {max_num_points}')
    print('Done')
