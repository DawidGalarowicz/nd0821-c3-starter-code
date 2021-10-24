import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def fetch_params(path):
    """Get parameters specified for DVC pipelines"""
    logger.info("Fetching parameters")
    with open(path, 'r') as fd:
        params = yaml.safe_load(fd)
    return params