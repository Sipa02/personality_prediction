"""
Author: Syifa Azzahro
Date: 18/8/2025
This is the pipeline.py module.
Usage:
- This module is used to initialize a TFX pipeline.
- It sets up the pipeline with the necessary components and configurations.
"""

from typing import Text, List

from absl import logging
# pylint: disable=import-error
from tfx.orchestration import metadata, pipeline


def init_pipeline(
    pipeline_root: Text,
    pipeline_name: str,
    metadata_path: str,
    components: List
) -> pipeline.Pipeline:
    """Initiate a TFX pipeline.

    Args:
        pipeline_root (Text): Path to the pipeline root directory.
        pipeline_name (str): Name of the pipeline.
        metadata_path (str): Path to the metadata directory.
        components (List): List of TFX components.

    Returns:
        pipeline.Pipeline: The configured TFX pipeline.
    """

    logging.info("Initializing pipeline: %s", pipeline_name)
    logging.info("Pipeline root set to: %s", pipeline_root)
    logging.info("Metadata path set to: %s", metadata_path)

    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path,
        ),
        beam_pipeline_args=beam_args,
    )
