from typing import Tuple

import numpy as np
from pandas import DataFrame, Series


def duplicates_from_value_counts(value_counts: Series) -> Tuple[int, float]:
    num_duplicate_rows = (value_counts > 1).sum()
    percentage = num_duplicate_rows / len(value_counts)
    return num_duplicate_rows, percentage


def find_duplicates_in_dataframes(source_df: DataFrame, synthetic_df: DataFrame):
    """ df1 is the source dataframe that df2 will be compared to """
    source_columns = source_df.columns.values
    synthetic_columns = synthetic_df.columns.values

    if not np.array_equal(source_columns, synthetic_columns):
        raise ValueError("Columns are not aligned!")

    source_unique = source_df.value_counts()
    synthetic_unique = synthetic_df.value_counts()

    num_source_dup, source_internal_dup_pct = duplicates_from_value_counts(source_unique)
    num_synthetic_dup, synthetic_internal_dup_pct = duplicates_from_value_counts(synthetic_unique)

    # Now that we found the inter-duplicated stats, reset the counts to 1 and sum.
    source_unique = source_unique.apply(lambda x: 1)
    synthetic_unique = synthetic_unique.apply(lambda x: 1)

    combined = source_unique.add(synthetic_unique, fill_value=0)
    dup_from_source = combined[(combined > 1)]
    source_dup_pct = len(dup_from_source) / len(source_unique)

    # Return the internal duplicates from the generated data and the percentage of unique entities that are the same.
    return num_source_dup, source_internal_dup_pct, num_synthetic_dup, synthetic_internal_dup_pct, len(dup_from_source), source_dup_pct
