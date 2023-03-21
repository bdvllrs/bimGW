import numpy as np

from bim_gw.datasets.data_module import split_indices_prop


def select_indices(all_indices, n):
    return np.random.choice(all_indices, n, replace=False)


def test_split_indices_prop_increasing_consistency_equal_sub():
    all_indices = np.arange(500_000)
    selected_indices = set(select_indices(all_indices, 200_000))
    np.random.seed(0)
    selected_low, rest_low = split_indices_prop(
        all_indices, selected_indices, prop=0.4
    )
    selected_sub_low, rest_sub_low = split_indices_prop(
        all_indices, selected_indices, prop=0.1
    )
    np.random.seed(0)
    selected_high, rest_high = split_indices_prop(
        all_indices, selected_indices, prop=0.7
    )
    selected_sub_high, rest_sub_high = split_indices_prop(
        all_indices, selected_indices, prop=0.1
    )
    # all elements of low should be in high
    assert len(np.setdiff1d(selected_low, selected_high)) == 0
    # sub_low should be equal to sub_high
    assert len(np.setdiff1d(selected_sub_low, selected_sub_high)) == 0
    assert len(np.setdiff1d(selected_sub_high, selected_sub_low)) == 0


def test_split_indices_prop_increasing_consistency_increasing_sub():
    all_indices = np.arange(500_000)
    selected_indices = set(select_indices(all_indices, 200_000))
    np.random.seed(0)
    selected_low, rest_low = split_indices_prop(
        all_indices, selected_indices, prop=0.4
    )
    selected_sub_low, rest_sub_low = split_indices_prop(
        all_indices, selected_indices, prop=0.2
    )
    np.random.seed(0)
    selected_high, rest_high = split_indices_prop(
        all_indices, selected_indices, prop=0.7
    )
    selected_sub_high, rest_sub_high = split_indices_prop(
        all_indices, selected_indices, prop=0.1
    )
    # all elements of low should be in high
    assert len(np.setdiff1d(selected_low, selected_high)) == 0
    # sub_low should be equal to sub_high
    assert len(np.setdiff1d(selected_sub_high, selected_sub_low)) == 0
