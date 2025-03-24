'''Template for the two classes hosts should customize for each competition.'''

import abc
import os
import time
import sys
import traceback
import warnings

from typing import Callable, Generator, Tuple

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.relay

_initial_import_time = time.time()
_issued_startup_time_warning = False


class Gateway(kaggle_evaluation.core.base_gateway.BaseGateway, abc.ABC):
    '''
    Template to start with when writing a new gateway.
    In most cases, hosts should only need to write get_all_predictions.
    There are two main methods for sending data to the inference_server hosts should understand:
    - Small datasets: use `self.predict`. Competitors will receive the data passed to self.predict as
    Python objects in memory. This is just a wrapper for self.client.send(); you can write additional
    wrappers if necessary.
    - Large datasets: it's much faster to send data via self.share_files, which is equivalent to making
    files available via symlink. See base_gateway.BaseGateway.share_files for the full details.
    '''
    @abc.abstractmethod
    def unpack_data_paths(self):
        ''' Map the contents of self.data_paths to the competition-specific entries
        Each competition should respect these paths to make it easy for competitors to
        run tests on their local machines or with custom files.

        Should include default paths to support data_paths = None.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def generate_data_batches(self) -> Generator:
        ''' Used by the default implementation of `get_all_predictions` so we can
        ensure `validate_prediction_batch` is run every time `predict` is called.

        This method must yield both the batch of data to be sent to `predict` and the validation
        data sent to `validate_prediction_batch`.
        '''
        raise NotImplementedError

    def get_all_predictions(self):
        all_predictions = []
        for data_batch, validation_batch in self.generate_data_batches():
            predictions = self.predict(*data_batch)
            self.validate_prediction_batch(predictions, validation_batch)
            all_predictions.append(predictions)
        return all_predictions

    def predict(self, *args, **kwargs):
        ''' self.predict will send all data in args and kwargs to the user container, and
        instruct the user container to generate a `predict` response.

        '''
        try:
            return self.client.send('predict', *args, **kwargs)
        except Exception as e:
            self.handle_server_error(e, 'predict')

    def set_response_timeout_seconds(self, timeout_seconds: float=6_000):
        # Set a response deadline that will apply after the very first repsonse
        self.client.endpoint_deadline_seconds = timeout_seconds

    def run(self):
        error = None
        try:
            self.unpack_data_paths()
            predictions = self.get_all_predictions()
            self.write_submission(predictions)
        except kaggle_evaluation.core.base_gateway.GatewayRuntimeError as gre:
            error = gre
        except Exception:
            # Get the full stack trace
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

            error = kaggle_evaluation.core.base_gateway.GatewayRuntimeError(
                kaggle_evaluation.core.base_gateway.GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION,
                error_str
            )

        self.client.close()
        if self.server:
            self.server.stop(0)

        if kaggle_evaluation.core.base_gateway.IS_RERUN:
            self.write_result(error)
        elif error:
            # For local testing
            raise error


class InferenceServer(abc.ABC):
    '''
    Base class for competition participants to inherit from when writing their submission. In most cases, users should
    only need to implement a `predict` function or other endpoints to pass to this class's constructor, and hosts will
    provide a mock Gateway for testing.
    '''
    def __init__(self, endpoint_listeners: Tuple[Callable]):
        self.server = kaggle_evaluation.core.relay.define_server(endpoint_listeners)
        self.client = None  # The inference_server can have a client but it isn't typically necessary.
        self._issued_startup_time_warning = False
        self._startup_limit_seconds = kaggle_evaluation.core.relay.STARTUP_LIMIT_SECONDS

    def serve(self):
        self.server.start()
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None:
            self.server.wait_for_termination()  # This will block all other code

    @abc.abstractmethod
    def _get_gateway_for_test(self, data_paths):
        # TODO: This should return a version of the competition-specific gateway that's able to load
        # data used for unit tests.
        raise NotImplementedError

    def run_local_gateway(self, data_paths=None):
        global _issued_startup_time_warning
        script_elapsed_seconds = time.time() - _initial_import_time
        if script_elapsed_seconds > self._startup_limit_seconds and not _issued_startup_time_warning:
            warnings.warn(
                f'''{int(script_elapsed_seconds)} seconds elapsed before server startup.
                This exceeds the startup time limit of {int(self._startup_limit_seconds)} seconds that the gateway will enforce
                during the rerun on the hidden test set. Start the server before performing any time consuming steps.''',
                category=RuntimeWarning
            )
            _issued_startup_time_warning = True

        self.server.start()
        try:
            self.gateway = self._get_gateway_for_test(data_paths)
            self.gateway.run()
        except Exception as err:
            raise err from None
        finally:
            self.server.stop(0)
