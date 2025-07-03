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
    
    weights = tl.extract_weights(dir, name='bitbrain')
    logger.info(f"Training data class weights:\n{weights}")

    stats = tl.get_stats(dir, name='bitbrain')
    logger.info(f"Calculated statistics from training data.")

    for data, process in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
        tl.standard_normalize(dir=dir, 
                              name='bitbrain', 
                              process=process,
                              include=['HB_1', 'HB_2', 'time'], 
                              stats=stats)

if __name__ == "__main__":
    main()