import torch

def update_model_params(model_file):
    params = torch.load(model_file)
    new_dict = {}
    for k in params.keys():
        new_key = k.replace('module.', '')
        new_dict[new_key] = params[k]
        print('{:s}, {:s}'.format(k, new_key))
    torch.save(new_dict, model_file + '.new')
    pass


if __name__=='__main__':
    update_model_params('./log/ep19.pkl')