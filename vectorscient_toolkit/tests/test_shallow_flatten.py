from ..utils import shallow_flatten


def test_flattening_nested_list():
    obj = {
        "key1": "atomic1",
        "key2": "atomic2",
        "actions": [
            {"type": "click", "url": "url1"},
            {"type": "click", "url": "url2"}
        ]
    }
    expected = [
        {
            "key1": "atomic1",
            "key2": "atomic2",
            "actions_type": "click",
            "actions_url": "url1"
        },
        {
            "key1": "atomic1",
            "key2": "atomic2",
            "actions_type": "click",
            "actions_url": "url2"
        }
    ]
    actual = shallow_flatten(obj)

    assert len(expected) == len(actual), "Wrong number of records"
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]


def test_flattening_nested_dict():
    obj = {
        "key1": "atomic1",
        "key2": {
            "nested1": "atomic2",
            "nested2": "atomic3"
        }
    }
    expected = [
        {
            "key1": "atomic1",
            "key2_nested1": "atomic2",
            "key2_nested2": "atomic3"
        }
    ]
    actual = shallow_flatten(obj)

    assert len(expected) == len(actual), "Wrong number of records"
    assert expected[0] == actual[0]


def test_flattening_several_nested_items():
    obj = {
        "key1": {
            "nested1": "atomic1"
        },
        "key2": "atomic2",
        "actions": [
            {"type": "bought", "url": "domain.com"},
            {"type": "didn't buy", "url": "example.org"}
        ],
        "players": [
            {"player_type": "flash"},
            {"player_type": "silverlight"}
        ],
        "key3": {
            "nested2": "atomic3",
            "nested3": "atomic4"
        }
    }
    expected = [
        {
            "key1_nested1": "atomic1",
            "key2": "atomic2",
            "key3_nested2": "atomic3",
            "key3_nested3": "atomic4",
            "actions_type": "bought",
            "actions_url": "domain.com",
            "players_player_type": "flash"
        },
        {
            "key1_nested1": "atomic1",
            "key2": "atomic2",
            "key3_nested2": "atomic3",
            "key3_nested3": "atomic4",
            "actions_type": "didn't buy",
            "actions_url": "example.org",
            "players_player_type": "flash"
        },
        {
            "key1_nested1": "atomic1",
            "key2": "atomic2",
            "key3_nested2": "atomic3",
            "key3_nested3": "atomic4",
            "actions_type": "bought",
            "actions_url": "domain.com",
            "players_player_type": "silverlight"
        },
        {
            "key1_nested1": "atomic1",
            "key2": "atomic2",
            "key3_nested2": "atomic3",
            "key3_nested3": "atomic4",
            "actions_type": "didn't buy",
            "actions_url": "example.org",
            "players_player_type": "silverlight"
        },
    ]
    actual = shallow_flatten(obj)

    assert len(expected) == len(actual), "Wrong number of records"
    for expected_item, actual_item in zip(expected, actual):
        assert len(expected_item) == len(actual_item)
        assert set(expected_item) == set(actual_item)
        assert set(expected_item.values()) == set(actual_item.values())



