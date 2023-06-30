import argparse


class TrainOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing,
    and saving the options. It also gathers additional options defined in
    <modify_commandline_options> functions
    in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both
        training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str,
                            default='./datasets/ \
                            preprocessed_hisab_ner_text.csv')
        parser.add_argument('--pred_text_path', type=str,
                            default='./datasets/pred_text.txt',
                            help='path to prediction text.')

        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='../output/checkpoints/hisab_ner.pth')
        # model parameters
        parser.add_argument('--model_name', type=str, default='BanglaBert',
                            help='chooses which model to use between \
                            [ BanglaBert | BasicBert')
        # dataset parameters
        parser.add_argument('--n_epochs', type=int, default=5,
                            help='number of epochs traininig')
        parser.add_argument('--num_workers', default=4,
                            type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int,
                            default=2, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int,
                            default=float("inf"),
                            help='Maximum number of samples allowed \
                                per dataset. If the dataset directory \
                                contains more than max_dataset_size, \
                                only a subset is loaded.')
        parser.add_argument('--num_labels', type=int,
                            default=7, help='number of unique labels')
        parser.add_argument('--max_token_length', type=int, default=512,
                            help='Maximum number of samples allowed per \
                            dataset. If the dataset directory contains \
                            more than max_dataset_size, only a subset \
                                is loaded.')
        # parser.add_argument('--preprocess', type=str, default=
        # 'resize_and_crop', help='scaling and cropping of images
        # at load time [resize_and_crop | crop | scale_width |
        # scale_width_and_crop | none]')
        parser.add_argument('--verbose', action='store_true',
                            help='if specified, print more \
                                debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + \
                                suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, 
        and set up gpu device."""
        opt = self.gather_options()
        # opt.isTrain = self.isTrain   # train or test

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
