import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Verifiable Networks', prog='octopus')
        
    parser.add_argument('configs', type=str,
                        help='Configurations file.')
    parser.add_argument('task', type=str,
                        choices=['train', 'verify', 'analyze', 'all'],
                        help='Select tasks to perform.')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='Random seed.')
    parser.add_argument('--result_dir', type=str,
                        default='./results/',
                        help='Result directory.')
    parser.add_argument('--platform', type=str,
                        default='local',
                        choices=['local', 'slurm'],
                        help='Platform to run jobs.')
    parser.add_argument('--override', action='store_true',
                        help='Overrides training/verification tasks.')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug log.')
    parser.add_argument('--dumb', action='store_true',
                        help='Silent mode.')

    return parser.parse_args()



# Training settings
def _parse_args():
    parser.add_argument('--problem', type=str, default='MNIST', choices=['MNIST','FashionMNIST'],
                        help='Which DL problem to run?')
    parser.add_argument('--net-name', type=str, default='NetS',
                        choices=['NetS','NetM','NetL','MnistConv'],
                        help='network name')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-pre-activation-value', action='store_true', default=False,
                        help='For Saving the Pre activation Values')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='default reuslt path for model')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='default reuslt path for logging')
    parser.add_argument('--meta_path', type=str, default='./meta',
                        help='default reuslt path for metadata')
    parser.add_argument('--figure_path', type=str, default='./figures',
                        help='default reuslt path for figures')
    parser.add_argument('--directory', type=str, default='directory',
                        help='directory containing pruning data')
    parser.add_argument('--adv_train', type=str, default='nature',
                        help='adversarial training option')
    #pruning params 
    parser.add_argument('--pr_ratio', type=float, default=0.2,
                        help='pruning ratio : (default: 0.2)')

    # rurh params
    parser.add_argument('--rurh', type=str, default=None,choices=[None, 'basic', 'upbd', 'ral'],
                        help='Inject the (r)educing (u)nsafe (r)eLU (h)euristic.\
                            Strategy of rurh 1) None: ... 2) upbd: upper bound 3) ral: relative accuracy loss')
    parser.add_argument('--rurh_ocrc', type=float, default=None,
                        help='rurh occurance(%%).')
    parser.add_argument('--rurh_itst', type=int, default=None,
                        help='rurh intensity(%%).')
    parser.add_argument('--rurh_upbd', type=int, default=None,
                        help='rurh upper bound(%%).')
    parser.add_argument('--rurh_ral', type=float, default=None,
                        help='rurh relative accuracy loss(%%).')
    parser.add_argument('--rurh_deactive_pre', type=int, default=1,
                        help='deactive rurh for the starting n epochs')
    parser.add_argument('--rurh_deactive_post', type=int, default=1,
                        help='deactive rurh for the ending n epochs')


    
    # rs loss
    parser.add_argument('--rs', type=str, default=None, choices=[None, 'basic'],
                        help='ReLU Stable loss.')
    return parser.parse_args()