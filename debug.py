from lib.lib_data import DataProcessor, ImageDataset
from lib.util import save_dict, load_dict
from tqdm import tqdm

# dp = DataProcessor()
# dp.fuse_bothsize_image('train')
# dp.get_superlabel_binary()
# dp.check_exist('val.json')

# ds = ImageDataset('train', is_transform=True)
# for item in tqdm(ds):
#     pass

def cover_dict():
    src = './log/ensemble/ep11.logit.test.pkl'
    dst = './log/ensemble/ep11.test.ens100.logit.pkl'
    cur_dict= load_dict(src)
    new_dict = {}
    for k,v in tqdm(cur_dict.items()):
        new_dict[k] = {'logit': v, 'label': -1}
    save_dict(new_dict, dst)

cover_dict()
pass