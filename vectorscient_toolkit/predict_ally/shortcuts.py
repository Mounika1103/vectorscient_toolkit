"""
Shortcuts to invoke prediction engine for most common scenarios.

The engine and it's components can be used directly, but the following functions
provide preconfigured settings that allow to easily invoke engine for
opportunity classification and reporting.
"""
from datetime import datetime
import time

import attrdict

from .clustering.user_config import TRIAL_USER_CONFIGURATION
from .clustering.unsupervised import OpportunityClassifier
from .reporting import ArchiveContext, PDFContext
from ..sources.data import CsvDataSource
from ..utils import merging_update, res
from .engine import PredictAlly


__all__ = ['run_opportunity_clustering_for_trial_user',
           'run_opportunity_clustering_for_registered_user']


def run_opportunity_clustering_for_trial_user(csv_file: str, config=None):
    """
    Runs prediction pipeline for a trial user.

    Args:
        csv_file (str): A path to uploaded CSV file with data to be clustered.
        config (dict): A pipeline clustering dictionary to override
            default config.

    Returns:
        output_file (str): The path to created file.
    """
    default_config = TRIAL_USER_CONFIGURATION.copy()
    default_config["records"]["source_file"] = csv_file
    config = merging_update(default_config, config)

    user = attrdict.AttrDict({'account': 'trial', 'parameters': config})
    engine = PredictAlly(user, save_report=False)

    source = CsvDataSource(file_name=csv_file)
    engine.load_data(source)

    classifier = OpportunityClassifier(grouping=False)
    engine.attach_technique(classifier)
    engine.run()

    output_file = config['reporting']['file_name']
    contexts = config['reporting']['context']
    _report_results_with_context(engine, output_file, contexts)

    return output_file


def run_opportunity_clustering_for_registered_user(config_file: str,
                                                   output_file=None,
                                                   context=None):
    """
    Runs prediction pipeline for a registered users, i.e. users with persistent
    configuration files.

    Args:
        config_file (str): A path to the user's configuration file.
        output_file (str): A path to the output file.
        context (str or List[str]): A reporting context name.

    Returns:
        output_file (str): The path to created file.
    """
    user = attrdict.AttrDict({'account': 'registered',
                              'parameters': {'config_file': config_file}})
    engine = PredictAlly(user, save_report=False)
    classifier = OpportunityClassifier()
    engine.attach_technique(classifier)
    engine.load_data_from_config()
    engine.run()

    contexts = [context] if isinstance(context, str) else context
    _report_results_with_context(engine, output_file, contexts)

    return output_file


def _report_results_with_context(engine, output_file, contexts):
    factories = {
        'archive': ArchiveContext,
        'pdf': PDFContext
    }

    for context_name in contexts:

        if context_name is None:
            context_name = 'archive'

        elif context_name not in factories:
            raise ValueError(
                "Unexpected context name: '{}'. Available choices "
                "are: {}".format(context_name, tuple(factories.keys())))

        filename = res("{}_disclaimer.html".format(engine.user_config.account))
        with open(filename) as fp:
            heading = fp.read()
        context = factories[context_name](disclaimer=heading)
        engine.report(context)

        if output_file is None:
            ts = datetime.fromtimestamp(time.time())
            prefix = "VS_Prediction_Output_"
            suffix = ts.strftime('%Y_%m_%d_%H_%M_%S')
            output_file = prefix + suffix

        context.save(file_name=output_file)
