"""Gateway notebook for https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2"""

import os
from typing import Generator

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates
import polars as pl


class AIMO2Gateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.set_response_timeout_seconds(60 * 30)  # 30 minutes

    def unpack_data_paths(self):
        if not self.data_paths:
            self.test_path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv'
        else:
            self.test_path = self.data_paths[0]

    def generate_data_batches(
        self,
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
        # Generate a random seed from system entropy
        random_seed = int.from_bytes(os.urandom(4), byteorder='big')

        # Read the test set and shuffle
        test = pl.read_csv(self.test_path).sample(
            fraction=1.0, shuffle=True, with_replacement=False, seed=random_seed
        )

        for row in test.iter_slices(n_rows=1):
            # Generate a problem instance and the validation id
            yield row, row.select('id')


if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = AIMO2Gateway()
        # Relies on valid default data paths
        gateway.run()
    else:
        print('Skipping run for now')
