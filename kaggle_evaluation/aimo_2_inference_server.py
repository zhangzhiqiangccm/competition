
import kaggle_evaluation.core.templates

import aimo_2_gateway


class AIMO2InferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None):
        return aimo_2_gateway.AIMO2Gateway(data_paths)
