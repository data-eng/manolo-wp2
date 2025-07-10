import os

from manolo.data.feature_extraction import extract_features
from manolo.data.feature_eval import eval_features
from manolo.base.utils.feat_utils import parser_function

# read and update args if needed
args, unparsed = parser_function()
args.save_root = os.path.join(args.save_root, args.note)

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(args.train_feat_file), exist_ok=True)

# run feature extraction
extract_features(args)
eval_features(args)