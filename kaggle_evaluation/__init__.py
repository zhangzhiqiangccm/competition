'''
Module implementing generic communication patterns with Python in / Python out
supporting many (nested) primitives + special data science types like DataFrames
or np.ndarrays, with gRPC + protobuf as a backing implementation.
'''

import os
import sys

# Provide additional import management since grpc_tools.protoc doesn't support relative imports
module_dir = os.path.dirname(os.path.abspath(__file__))
gen_dir = os.path.join(module_dir, 'core', 'generated')

if not os.path.exists(os.path.join(gen_dir, 'kaggle_evaluation_pb2.py')):
    print('kaggle evaluation proto and gRPC generated files are missing')
    sys.exit(1)

sys.path.append(module_dir)
sys.path.append(gen_dir)


__version__ = '0.3.0'
