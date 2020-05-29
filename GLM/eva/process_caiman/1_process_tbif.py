import struct
from collections import namedtuple

import numpy as np

Header = namedtuple('Header', [
    'framesPerPulse',
    'secsPerFrame',
    'w',
    'h',
    'Vxmin',
    'Vxmax',
    'Vymin',
    'Vymax',
    'minVal',
    'maxVal'
])

Field = namedtuple('Field', ['n', 'type_'])

# Parameters
filename = "08-17-14_1437_F1_6dpfCOMPLETESET_WB_overclimbing_z-1.tbif"
save_path = "08-17-14_1437_F1_6dpfCOMPLETESET_WB_overclimbing_z-1.npz"
header_size = 48
header_type = "=IdHHffffdd"
data_format = {
    # Name: (number of variables, type)
    'zpos': Field(1, np.float32),
    'stim': Field(3, np.float32),
    'img': Field(None, np.uint16)
}

# Load file
with open(filename, mode='rb') as f:
    # Get header
    header = f.read(header_size)
    header = Header(*struct.unpack(header_type, header))
    print(header)

    # Convert to np.ndarray for efficient reshaping.
    raw = f.read()
    raw = np.array(bytearray(raw), dtype=np.byte)

img_size = header.w * header.h
data_format['img'] = Field(img_size, data_format['img'].type_)
field_bytes = np.array([0] + [t.n * np.dtype(t.type_).itemsize for t in data_format.values()])

# Check if data format is correct.
n = len(raw) / np.sum(field_bytes)
if round(n) == n:
    n = int(n)
else:
    raise ValueError('Number of frames is not integer.')
print(f'{n} frames')

# Convert byte into specified format.
raw = np.reshape(raw, (n, np.sum(field_bytes)))
bytes_sum = np.cumsum(field_bytes)
for i, (name, value) in enumerate(data_format.items()):
    temp = np.ascontiguousarray(raw[:, bytes_sum[i]: bytes_sum[i + 1]])
    temp = np.frombuffer(temp, dtype=value.type_)
    globals()[name] = np.reshape(temp, (n, value.n))
    print(f'Processed {name}.')
del raw

# Reshape img into W*H
img = np.reshape(img, (n, header.h, header.w))

print(f'Writing to {save_path}.')
np.savez(save_path, **{k: globals()[k] for k in data_format.keys()})
np.savetxt('../stim_data.txt', stim)
np.savetxt('../mean.txt', np.mean(img, axis=0))
print('OK')

# %%

from skimage.external.tifffile import imsave

imsave(save_path + '.tif', img)
