import os
import numpy as np
import sys



d = {'test acc label': [],
     'test acc sens':  [],
     'test acc label deter': [],
     'test acc sens deter': []}

model_name = sys.argv[1]
dataset_name = sys.argv[2]

directory = 'trained_models/' + model_name + '/' + dataset_name + '/'

file_num = 0
for filename in os.listdir(directory):
    file_path = directory + filename + '/evaluation.txt'
    print(file_path)
    with open(file_path) as f:
        for line in f:
            (key, val) = line.split(':')
            d[key].append(float(val))

    file_num += 1

for key in d.keys():
    d[key] = np.array(d[key])

with open('trained_models/'+directory.replace('/', '-')[15:-1] + '.txt', 'w') as file:
    file.write('Average of {} runs\n'.format(file_num))
    for key in d.keys():
        file.write(key + ' mean: ' + str(np.mean(d[key])) + ' std: ' + str(np.std(d[key])) +'\n')
    file.close()

