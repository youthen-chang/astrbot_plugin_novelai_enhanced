import pytest
from enhanced_test.generation import Rejected
from enhanced_test.prompts import split_extra_negative


@pytest.mark.parametrize(
    "text,positive,negative",
    [
        (
            "示例角色，风格040，竖 -n:glasses, hat。",
            "示例角色，风格040，竖",
            "glasses, hat",
        ),
        ("1girl -N：glasses, hat.", "1girl", "glasses, hat"),
        ("1girl\n-n：glasses。", "1girl", "glasses"),
        ("1girl -n:1.5::bad_hands::。 横", "1girl, 横", "1.5::bad_hands::"),
        ("-N:hat. 花田", "花田", "hat"),
        ("1girl-n:glasses。", "1girl", "glasses"),
    ],
)
def test_split(text, positive, negative):
    assert split_extra_negative(text) == (positive, negative, True)


def test_no_marker_does_not_reinterpret_description():
    assert split_extra_negative("negative space, 1girl") == (
        "negative space, 1girl",
        "",
        False,
    )
    assert split_extra_negative("讨论负面提示词，少女") == (
        "讨论负面提示词，少女",
        "",
        False,
    )


def test_explicit_blank_can_clear_previous_extra_negative():
    assert split_extra_negative("-n:。") == ("", "", True)
    assert split_extra_negative("") == ("", "", False)


def test_duplicate_markers_rejected():
    with pytest.raises(Rejected):
        split_extra_negative("1girl -n:hat。 -N：glasses.")


def test_missing_period_rejected():
    with pytest.raises(Rejected, match="句号"):
        split_extra_negative("1girl -n:glasses")
