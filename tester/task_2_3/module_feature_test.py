import os

from manolo.data.data_synthesis.feature_extraction import extract_features
from manolo.data.data_synthesis.feature_eval import eval_features
from manolo.base.utils.feat_utils import parser_function

# read and update args if needed
args, unparsed = parser_function()
args.save_root = os.path.join(args.save_root, args.note)

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(args.train_feat_file), exist_ok=True)

# run feature extraction
extract_features(args)  # saves extracted features in args.test_feat_file
eval_features(args)     # returns a dictionary with the evaluation results of the features in args.test_feat_file