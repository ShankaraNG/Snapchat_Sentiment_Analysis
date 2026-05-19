from ml_build.services.pipelinerunner import pipeline_runner
from ml_build.logger import get_logger
import os
import sys

log = get_logger('Main Runner')

def main():
    try:
        log.info('Starting the Model building and Training Pipeline')
        result = pipeline_runner()
        if result != 'successfull':
            raise Exception("Pipeline runner did not complete successfully")
        log.info('Model Training and TestingPipeline Ended')
    except Exception as e:
        log.exception("Pipeline failed with an error")
        sys.exit(1)

if __name__ == "__main__":
    main()