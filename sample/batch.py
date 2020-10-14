import numpy as np

from aicentro.loader.keras_loader import KerasLoader
from aicentro.batch.base_batch import BaseBatch


print('# Service Model Load...')
loader = KerasLoader(model_filename='iris-classification')
batch = BaseBatch(loader=loader)

print('# Find Input Files...')
inputs_files = batch.load_input_files()

print('# Input Files... {}'.format(inputs_files))

print('# Output File Init...')
batch.init_output_file('output.csv')

print('# Predict...')
for inp in batch.read_csv(inputs_files[0]):
    np_inp = inp.to_numpy()
    outputs = batch.loader.predict('inputs', np_inp)
    concat_outputs = np.concatenate((np_inp, outputs), axis=1)
    batch.write_output('output.csv', concat_outputs)

print('# Completed Batch...')

