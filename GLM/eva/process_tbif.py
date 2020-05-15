import struct
import numpy as np

from collections import namedtuple

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
filename = "09-04-14_1715_F1_compl_pyoga_ZOOM_2_z-1.tbif"
save_path = "09-04-14_1715_F1_compl_pyoga_ZOOM_2_z-1.npz"
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
print('OK')

#%%
from skimage.external.tifffile import imsave

imsave(save_path+'.tif', img)