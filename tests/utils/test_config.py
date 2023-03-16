from bim_gw.utils.config import get_argv_dotlist


def test_get_argv_main():
    dotlist = get_argv_dotlist(
        [
            "param1=1",
            "param3.param3_1=4",
        ]
    )
    assert dotlist == [
        "param1=1",
        "param3.param3_1=4",
    ]


def test_get_argv_with_spaces():
    dotlist = get_argv_dotlist(
        [
            "param2", "3",
            "param3.param3_2", "5",
            "param4 6",
            "param5 7 ",
        ]
    )

    assert dotlist == [
        "param2=3",
        "param3.param3_2=5",
        "param4=6",
        "param5=7"
    ]


def test_get_argv_with_flags():
    dotlist = get_argv_dotlist(
        [
            "param2", "3",
            "param3.param3_2", "5",
            "param4 6",
            "param5 7 ",
            "-d",
            "--dry-run",
        ]
    )

    assert dotlist == [
        "param2=3",
        "param3.param3_2=5",
        "param4=6",
        "param5=7",
        "-d=True",
        "--dry-run=True",
    ]


def test_get_argv_with_flags_no_dash():
    dotlist = get_argv_dotlist(
        [
            "param2", "3",
            "param3.param3_2", "5",
            "param4 6",
            "param5 7 ",
            "-d",
            "param6=8",
            "-p",  # failing case
            "param7", "9"
        ]
    )

    assert dotlist == [
        "param2=3",
        "param3.param3_2=5",
        "param4=6",
        "param5=7",
        "-d=True",
        "param6=8",
        "-p=param7",
        "9=True"
    ]
