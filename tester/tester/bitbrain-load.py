import os
import torch_loader as tl

logger = tl.get_logger(level='INFO')

def main():
    """
    Main function to create torch loaders from the Bitbrain dataset, suitable for machine learning tasks.
    """
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    dir = tl.get_dir(root, 'datasets', 'bitbrain_conv')

    logger.info("Shifting labels in the entire dataset.")
    tl.shift_labels(dir, name='bitbrain')

    logger.info("Splitting data into train, val, test.")
    train_data, val_data, test_data = tl.split_data(dir=dir, 
                                                    name='bitbrain', 
                                                    train_size=0.6, 
                                                    val_size=0.2, 
                                                    test_size=0.2,
                                                    exist=False)
    
    logger.info("Extracting class weights from training set.")
    weights = tl.extract_weights(dir, name='bitbrain')

    logger.info(f"Torch loaders created!")

if __name__ == "__main__":
    main()