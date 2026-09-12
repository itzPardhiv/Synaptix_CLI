from synaptix.research import parallel_map


def test_empty_parallel_map():

    result = parallel_map(
        lambda x: x * 2,
        [],
    )

    assert result == []


def test_zero_workers_are_safe():

    result = parallel_map(
        lambda x: x * 2,
        [1, 2, 3],
        workers=0,
    )

    assert sorted(result) == [
        2,
        4,
        6,
    ]