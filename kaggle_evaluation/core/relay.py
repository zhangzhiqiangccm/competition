'''
Core implementation of the client module, implementing generic communication
patterns with Python in / Python out supporting many (nested) primitives +
special data science types like DataFrames or np.ndarrays, with gRPC + protobuf
as a backing implementation.
'''

import grpc
import io
import json
import socket
import time

from concurrent import futures
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow

import kaggle_evaluation.core.generated.kaggle_evaluation_pb2 as kaggle_evaluation_proto
import kaggle_evaluation.core.generated.kaggle_evaluation_pb2_grpc as kaggle_evaluation_grpc


_SERVICE_CONFIG = {
    # Service config proto: https://github.com/grpc/grpc-proto/blob/ec886024c2f7b7f597ba89d5b7d60c3f94627b17/grpc/service_config/service_config.proto#L377
    'methodConfig': [
        {
            'name': [{}],  # Applies to all methods
            # See retry policy docs: https://grpc.io/docs/guides/retry/
            'retryPolicy': {
                'maxAttempts': 5,
                'initialBackoff': '0.1s',
                'maxBackoff': '1s',
                'backoffMultiplier': 1, # Ensure relatively rapid feedback in the event of a crash
                'retryableStatusCodes': ['UNAVAILABLE'],
            },
        }
    ]
}
_GRPC_PORT = 50051
_GRPC_CHANNEL_OPTIONS = [
    # -1 for unlimited message send/receive size
    # https://github.com/grpc/grpc/blob/v1.64.x/include/grpc/impl/channel_arg_names.h#L39
    ('grpc.max_send_message_length', -1),
    ('grpc.max_receive_message_length', -1),
    # https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    ('grpc.keepalive_time_ms', 60_000),  # Time between heartbeat pings
    ('grpc.keepalive_timeout_ms', 5_000),  # Time allowed to respond to pings
    ('grpc.http2.max_pings_without_data', 0), # Remove another cap on pings
    ('grpc.keepalive_permit_without_calls', 1), # Allow heartbeat pings at any time
    ('grpc.http2.min_ping_interval_without_data_ms', 1_000),
    ('grpc.service_config', json.dumps(_SERVICE_CONFIG)),
]


DEFAULT_DEADLINE_SECONDS = 60 * 60
_RETRY_SLEEP_SECONDS = 1
# Enforce a relatively strict server startup time so users can get feedback quickly if they're not
# configuring KaggleEvaluation correctly. We really don't want notebooks timing out after nine hours
# somebody forgot to start their inference_server. Slow steps like loading models
# can happen during the first inference call if necessary.
STARTUP_LIMIT_SECONDS = 60 * 15

### Utils shared by client and server for data transfer

# pl.Enum is currently unstable, but we should eventually consider supporting it.
# https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Enum.html#polars.datatypes.Enum
_POLARS_TYPE_DENYLIST = set([pl.Enum, pl.Object, pl.Unknown])

def _serialize(data) -> kaggle_evaluation_proto.Payload:
    '''Maps input data of one of several allow-listed types to a protobuf message to be sent over gRPC.

    Args:
        data: The input data to be mapped. Any of the types listed below are accepted.

    Returns:
        The Payload protobuf message.

    Raises:
        TypeError if data is of an unsupported type.
    '''
    # Python primitives and Numpy scalars
    if isinstance(data, np.generic):
        # Numpy functions that return a single number return numpy scalars instead of python primitives.
        # In some cases this difference matters: https://numpy.org/devdocs/release/2.0.0-notes.html#representation-of-numpy-scalars-changed
        # Ex: np.mean(1,2) yields np.float64(1.5) instead of 1.5.
        # Check for numpy scalars first since most of them also inherit from python primitives.
        # For example, `np.float64(1.5)` is an instance of `float` among many other things.
        # https://numpy.org/doc/stable/reference/arrays.scalars.html
        assert data.shape == ()  # Additional validation that the np.generic type remains solely for scalars
        assert isinstance(data, np.number) or isinstance(data, np.bool_)  # No support for bytes, strings, objects, etc
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return kaggle_evaluation_proto.Payload(numpy_scalar_value=buffer.getvalue())
    elif isinstance(data, str):
        return kaggle_evaluation_proto.Payload(str_value=data)
    elif isinstance(data, bool): # bool is a subclass of int, so check that first
        return kaggle_evaluation_proto.Payload(bool_value=data)
    elif isinstance(data, int):
        return kaggle_evaluation_proto.Payload(int_value=data)
    elif isinstance(data, float):
        return kaggle_evaluation_proto.Payload(float_value=data)
    elif data is None:
        return kaggle_evaluation_proto.Payload(none_value=True)
    # Iterables for nested types
    if isinstance(data, list):
        return kaggle_evaluation_proto.Payload(list_value=kaggle_evaluation_proto.PayloadList(payloads=map(_serialize, data)))
    elif isinstance(data, tuple):
        return kaggle_evaluation_proto.Payload(tuple_value=kaggle_evaluation_proto.PayloadList(payloads=map(_serialize, data)))
    elif isinstance(data, dict):
        serialized_dict = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f'KaggleEvaluation only supports dicts with keys of type str, found {type(key)}.')
            serialized_dict[key] = _serialize(value)
        return kaggle_evaluation_proto.Payload(dict_value=kaggle_evaluation_proto.PayloadMap(payload_map=serialized_dict))
    # Allowlisted special types
    if isinstance(data, pd.DataFrame):
        buffer = io.BytesIO()
        data.to_parquet(buffer, index=False, compression='lz4')
        return kaggle_evaluation_proto.Payload(pandas_dataframe_value=buffer.getvalue())
    elif isinstance(data, pl.DataFrame):
        data_types = set(i.base_type() for i in data.dtypes)
        banned_types = _POLARS_TYPE_DENYLIST.intersection(data_types)
        if len(banned_types) > 0:
            raise TypeError(f'Unsupported Polars data type(s): {banned_types}')

        table = data.to_arrow()
        buffer = io.BytesIO()
        with pyarrow.ipc.new_stream(buffer, table.schema, options=pyarrow.ipc.IpcWriteOptions(compression='lz4')) as writer:
            writer.write_table(table)
        return kaggle_evaluation_proto.Payload(polars_dataframe_value=buffer.getvalue())
    elif isinstance(data, pd.Series):
        buffer = io.BytesIO()
        # Can't serialize a pd.Series directly to parquet, must use intermediate DataFrame
        pd.DataFrame(data).to_parquet(buffer, index=False, compression='lz4')
        return kaggle_evaluation_proto.Payload(pandas_series_value=buffer.getvalue())
    elif isinstance(data, pl.Series):
        buffer = io.BytesIO()
        # Can't serialize a pl.Series directly to parquet, must use intermediate DataFrame
        pl.DataFrame(data).write_parquet(buffer, compression='lz4', statistics=False)
        return kaggle_evaluation_proto.Payload(polars_series_value=buffer.getvalue())
    elif isinstance(data, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return kaggle_evaluation_proto.Payload(numpy_array_value=buffer.getvalue())
    elif isinstance(data, io.BytesIO):
        return kaggle_evaluation_proto.Payload(bytes_io_value=data.getvalue())

    raise TypeError(f'Type {type(data)} not supported for KaggleEvaluation.')


def _deserialize(payload: kaggle_evaluation_proto.Payload):
    '''Maps a Payload protobuf message to a value of whichever type was set on the message.

    Args:
        payload: The message to be mapped.

    Returns:
        A value of one of several allow-listed types.

    Raises:
        TypeError if an unexpected value data type is found.
    '''
    # Primitives
    if payload.WhichOneof('value') == 'str_value':
        return payload.str_value
    elif payload.WhichOneof('value') == 'bool_value':
        return payload.bool_value
    elif payload.WhichOneof('value') == 'int_value':
        return payload.int_value
    elif payload.WhichOneof('value') == 'float_value':
        return payload.float_value
    elif payload.WhichOneof('value') == 'none_value':
        return None
    # Iterables for nested types
    elif payload.WhichOneof('value') == 'list_value':
        return list(map(_deserialize, payload.list_value.payloads))
    elif payload.WhichOneof('value') == 'tuple_value':
        return tuple(map(_deserialize, payload.tuple_value.payloads))
    elif payload.WhichOneof('value') == 'dict_value':
        return {key: _deserialize(value) for key, value in payload.dict_value.payload_map.items()}
    # Allowlisted special types
    elif payload.WhichOneof('value') == 'pandas_dataframe_value':
        return pd.read_parquet(io.BytesIO(payload.pandas_dataframe_value))
    elif payload.WhichOneof('value') == 'polars_dataframe_value':
        with pyarrow.ipc.open_stream(payload.polars_dataframe_value) as reader:
            table = reader.read_all()
        return pl.from_arrow(table)
    elif payload.WhichOneof('value') == 'pandas_series_value':
        # Pandas will still read a single column csv as a DataFrame.
        df = pd.read_parquet(io.BytesIO(payload.pandas_series_value))
        return pd.Series(df[df.columns[0]])
    elif payload.WhichOneof('value') == 'polars_series_value':
        return pl.Series(pl.read_parquet(io.BytesIO(payload.polars_series_value)))
    elif payload.WhichOneof('value') == 'numpy_array_value':
        return np.load(io.BytesIO(payload.numpy_array_value), allow_pickle=False)
    elif payload.WhichOneof('value') == 'numpy_scalar_value':
        data = np.load(io.BytesIO(payload.numpy_scalar_value), allow_pickle=False)
        # As of Numpy 2.0.2, np.load for a numpy scalar yields a dimensionless array instead of a scalar
        data = data.dtype.type(data) # Restore the expected numpy scalar type.
        assert data.shape == ()  # Additional validation that the np.generic type remains solely for scalars
        assert isinstance(data, np.number) or isinstance(data, np.bool_)  # No support for bytes, strings, objects, etc
        return data
    elif payload.WhichOneof('value') == 'bytes_io_value':
        return io.BytesIO(payload.bytes_io_value)

    raise TypeError(f'Found unknown Payload case {payload.WhichOneof("value")}')

### Client code

class Client():
    '''
    Class which allows callers to make KaggleEvaluation requests.
    '''
    def __init__(self, channel_address: str='localhost'):
        self.channel_address = channel_address
        self.channel = grpc.insecure_channel(f'{channel_address}:{_GRPC_PORT}', options=_GRPC_CHANNEL_OPTIONS)
        self._made_first_connection = False
        self.endpoint_deadline_seconds = DEFAULT_DEADLINE_SECONDS
        self.stub = kaggle_evaluation_grpc.KaggleEvaluationServiceStub(self.channel)

    def _send_with_deadline(self, request):
        ''' Sends a message to the server while also:
        - Throwing an error as soon as the inference_server container has been shut down.
        - Setting a deadline of STARTUP_LIMIT_SECONDS for the inference_server to startup.
        '''
        if self._made_first_connection:
            return self.stub.Send(request, wait_for_ready=False, timeout=self.endpoint_deadline_seconds)

        first_call_time = time.time()
        # Allow time for the server to start as long as its container is running
        while time.time() - first_call_time < STARTUP_LIMIT_SECONDS:
            try:
                response = self.stub.Send(request, wait_for_ready=False)
                self._made_first_connection = True
                break
            except grpc._channel._InactiveRpcError as err:
                if 'StatusCode.UNAVAILABLE' not in str(err):
                    raise err
            # Confirm the inference_server container is still alive & it's worth waiting on the server.
            # If the inference_server container is no longer running this will throw a socket.gaierror.
            socket.gethostbyname(self.channel_address)
            time.sleep(_RETRY_SLEEP_SECONDS)

        if not self._made_first_connection:
            raise RuntimeError(f'Failed to connect to server after waiting {STARTUP_LIMIT_SECONDS} seconds')
        return response

    def send(self, name: str, *args, **kwargs):
        '''Sends a single KaggleEvaluation request.

        Args:
            name: The endpoint name for the request.
            *args: Variable-length/type arguments to be supplied on the request.
            **kwargs: Key-value arguments to be supplied on the request.

        Returns:
            The response, which is of one of several allow-listed data types.
        '''
        request = kaggle_evaluation_proto.KaggleEvaluationRequest(
                name=name,
                args=map(_serialize, args),
                kwargs={key: _serialize(value) for key, value in kwargs.items()}
        )
        response = self._send_with_deadline(request)

        return _deserialize(response.payload)

    def close(self):
        self.channel.close()


### Server code

class KaggleEvaluationServiceServicer(kaggle_evaluation_grpc.KaggleEvaluationServiceServicer):
    '''
    Class which allows serving responses to KaggleEvaluation requests. The inference_server will run this service to listen for and respond
    to requests from the Gateway. The Gateway may also listen for requests from the inference_server in some cases.
    '''
    def __init__(self, listeners: List[callable]):
        self.listeners_map = dict((func.__name__, func) for func in listeners)

    # pylint: disable=unused-argument
    def Send(self, request: kaggle_evaluation_proto.KaggleEvaluationRequest, context: grpc.ServicerContext) -> kaggle_evaluation_proto.KaggleEvaluationResponse:
        '''Handler for gRPC requests that deserializes arguments, calls a user-registered function for handling the
        requested endpoint, then serializes and returns the response.

        Args:
            request: The KaggleEvaluationRequest protobuf message.
            context: (Unused) gRPC context.

        Returns:
            The KaggleEvaluationResponse protobuf message.

        Raises:
            NotImplementedError if the caller has not registered a handler for the requested endpoint.
        '''
        if request.name not in self.listeners_map:
            raise NotImplementedError(f'No listener for {request.name} was registered.')

        args = map(_deserialize, request.args)
        kwargs = {key: _deserialize(value) for key, value in request.kwargs.items()}
        response_function = self.listeners_map[request.name]
        response_payload = _serialize(response_function(*args, **kwargs))
        return kaggle_evaluation_proto.KaggleEvaluationResponse(payload=response_payload)

def define_server(*endpoint_listeners: Tuple[Callable]) -> grpc.server:
    '''Registers the endpoints that the container is able to respond to, then starts a server which listens for
    those endpoints. The endpoints that need to be implemented will depend on the specific competition.

    Args:
        endpoint_listeners: Tuple of functions that define how requests to the endpoint of the function name should be
            handled.

    Returns:
        The gRPC server object, which has been started. It should be stopped at exit time.

    Raises:
        ValueError if parameter values are invalid.
    '''
    if not endpoint_listeners:
        raise ValueError('Must pass at least one endpoint listener, e.g. `predict`')
    for func in endpoint_listeners:
        if not isinstance(func, Callable):
            raise ValueError('Endpoint listeners passed to `serve` must be functions')
        if func.__name__ == '<lambda>':
            raise ValueError('Functions passed as endpoint listeners must be named')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=_GRPC_CHANNEL_OPTIONS)
    kaggle_evaluation_grpc.add_KaggleEvaluationServiceServicer_to_server(KaggleEvaluationServiceServicer(endpoint_listeners), server)
    server.add_insecure_port(f'[::]:{_GRPC_PORT}')
    return server
