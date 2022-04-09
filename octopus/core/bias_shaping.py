import numpy as np
import torch


def wd_loss(model, device):
    loss = torch.zeros(3).to(device=device)
    params = [14*14, 7*7, 1]
    c = 0
    for layer in model.state_dict():
        if 'weight' in layer and 'out' not in layer:
            w = model.state_dict()[layer]
            x = torch.abs(w)
            x = torch.sum(w) * params[c]
            loss[c] = x
            c += 1
    loss = torch.sum(loss)
    return loss


def bias_shaping(args, data, activation, model, device, epoch, batch_idx, test_accuracy, last_pre_rurh_accuracy):
    f_rurh = False

    safe_le_zero_all = []
    safe_ge_zero_all = []
    for layer in activation.keys():
        val = activation[layer].view(activation[layer].size(0),-1).numpy()
        val_min = np.min(val, axis=0)
        val_max = np.max(val, axis=0)
        safe_ge_zero = (np.asarray(val_min) >= 0).sum()
        safe_le_zero = (np.asarray(val_max) <= 0).sum()

        safe_le_zero_all += [safe_le_zero]
        safe_ge_zero_all += [safe_ge_zero]

    safe_all = sum(safe_le_zero_all)+sum(safe_ge_zero_all)
    max_safe_relu = sum([activation[layer].view(activation[layer].size(0),-1).shape[-1] for layer in activation])
    
    # When is RURH activated?
    # 1) don't happen on first mini batch
    # 1) rurh is on
    # 2) last epochs is excepted
    # 3) occurance > 0
    # 4) ocrc% of time
    # 5) doesnt exceed upper bound
    if batch_idx > 0\
        and args.rurh_itst is not None and args.rurh_ocrc is not None\
        and epoch > args.rurh_deactive_pre and epoch <= args.epochs-args.rurh_deactive_post\
        and args.rurh_itst > 0 and np.random.rand() > 1 - args.rurh_ocrc:
        
        if args.rurh is None:
            f_rurh = False
        elif args.rurh == 'basic':
            f_rurh = True
        elif args.rurh == 'upbd':
            assert args.upbd is not None
            f_rurh = safe_all/max_safe_relu * 100 < args.rurh_upbd
        elif args.rurh == 'ral':
            assert args.rurh_ral is not None
            if last_pre_rurh_accuracy is None:
                f_rurh = False
            else:
                f_rurh = (last_pre_rurh_accuracy - test_accuracy[-1])/last_pre_rurh_accuracy <= args.rurh_ral
        else:
            assert False

        if f_rurh:
            for layer in activation.keys():
                model.eval()
                model(data)

                val = activation[layer].view(activation[layer].size(0),-1).numpy()
                val_min = np.min(val, axis=0)
                val_max = np.max(val, axis=0)
                safe_ge_zero = (np.asarray(val_min) >= 0).sum()
                safe_le_zero = (np.asarray(val_max) <= 0).sum()

                # assert safe_ge_zero == 0
                val_min_lt_zero = np.copy(val_min)
                val_max_gt_zero = np.copy(val_max)
                
                # pick lb < 0
                val_min_lt_zero[val_min_lt_zero > 0] = 0
                # print(val_min_lt_zero)
                val_min_lt_zero*= -1
                # pick ub > 0
                val_max_gt_zero[val_max_gt_zero < 0] = 0
                # print(val_max_gt_zero)

                val_abs_min = np.min(np.array([val_min_lt_zero, val_max_gt_zero]), axis=0)
                len(np.where(val_abs_min==0)[0]) == safe_ge_zero + safe_le_zero

                n = int(len(np.where(val_abs_min!=0)[0]) * args.rurh_itst/100)
                pivot_value = val_abs_min[np.argsort(val_abs_min)[-n]]

                val_abs_min[val_abs_min < pivot_value] = 0

                a = np.where(val_min_lt_zero == val_abs_min)
                b = np.where(val_max_gt_zero == val_abs_min)
                x = np.zeros(val_min.shape)
                x[a[0]] = val_abs_min[a[0]]#+SHIFT_EPSILON
                x*=-1
                x[b[0]] = val_abs_min[b[0]]#+SHIFT_EPSILON
                x*=-1

                pretrained = model.state_dict()
                fc1_bias = pretrained[f'{layer}.bias']
                epsilon = 0.1
                new_bias = fc1_bias.detach().numpy() + x
                new_bias = torch.from_numpy(new_bias).to(device, dtype=torch.float32)
                # model.fc1.bias = model.fc1.bias - new_bias
                # model.fc1.bias = nn.Parameter(torch.randn(128))
                pretrained[f'{layer}.bias'] = new_bias
                model.load_state_dict(pretrained)

    if f_rurh:
        safe_le_zero_all = []
        safe_ge_zero_all = []
        for layer in activation.keys():
            val = activation[layer].view(activation[layer].size(0),-1).numpy()
            val_min = np.min(val, axis=0)
            val_max = np.max(val, axis=0)
            safe_ge_zero = (np.asarray(val_min) >= 0).sum()
            safe_le_zero = (np.asarray(val_max) <= 0).sum()

            safe_le_zero_all += [safe_le_zero]
            safe_ge_zero_all += [safe_ge_zero]
    
    return f_rurh, sum(safe_le_zero_all), sum(safe_ge_zero_all)