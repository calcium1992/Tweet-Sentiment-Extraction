import yaml
import logging
import argparse
from module import Preprocessor, Trainer, Predictor

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Process commandline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    # Logger
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)
    logger = logging.getLogger('global_logger')

    # Main
    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)

            # Preprocessing
            preprocessor = Preprocessor(config=config['preprocessing'], logger=logger)
            preprocessor.generate_data_loaders()

            # Training
            # trainer = Trainer(config=config['training'], logger=logger, preprocessor=preprocessor)
            # trainer.kfold_training()

            # Predicting
            predictor = Predictor(config=config['predict'], logger=logger, preprocessor=preprocessor)
            predictions = predictor.predict()
            print(predictions)
            # predictor.save_result(preprocessor.test_ids, y_prob_pred)
        except yaml.YAMLError as err:
            logger.warning(f'Config file err: {err}')