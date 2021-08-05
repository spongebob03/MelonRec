from sklearn.utils import resample
from arena_util2 import load_json, write_json

def load_origin_data(key):
    PATH = 'res/'
    json_PATH = PATH + key + '.json'
    try:
        return load_json(fname=json_PATH)
    except FileNotFoundError:
        return f'key: {key}, 파일 이름이 존재하지 않습니다'

def resample_data(origin_data, random_seed, zip_size=100):
    return resample(
        origin_data,
        replace=False,
        n_samples=len(origin_data) // zip_size,
        random_state=SEED
    )

origin_train = load_origin_data('train')
origin_test = load_origin_data('test')
origin_val = load_origin_data('val')

SEED = 42
ZIP_SIZE = 100

small_train = resample_data(origin_train, SEED, ZIP_SIZE)
small_test = resample_data(origin_test, SEED, ZIP_SIZE)
small_val = resample_data(origin_val, SEED, ZIP_SIZE)

print(f'origin size: (train: {len(origin_train)}), (test: {len(origin_test)}), (val: {len(origin_val)})')
print(f'origin size: (train: {len(small_train)}), (test: {len(small_test)}), (val: {len(small_val)})')

write_json(small_train, './small_datasets/train.json')
write_json(small_test, './small_datasets/test.json')
write_json(small_val, './small_datasets/val.json')