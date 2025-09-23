import os

from manolo.base.metrics.code_carbon_utils import codecarbon_init
codecarbon_init(cc_path="./output_codecarbon/", deactivate_codecarbon=True)

from manolo.data.synth.feature_extraction import extract_features
from manolo.data.synth.feature_eval import eval_features
from manolo.data.synth.feature_extraction_utils import parser_function

# read and update args if needed
args, unparsed = parser_function()
args.save_root = os.path.join(args.save_root, args.exp_name)

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(args.train_feat_file), exist_ok=True)

args.select_cuda = -1

# run feature extraction
extract_features(args)  # saves extracted features in args.test_feat_file
eval_features(args)     # returns a dictionary with the evaluation results of the features in args.test_feat_file