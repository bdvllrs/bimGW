from pathlib import Path
from unittest import mock

from auto_sbatch import SlurmScriptParser
from omegaconf import OmegaConf

from bim_gw.utils import get_args
from slurm.launch import grid_search_exclusion_from_past_search, main as launch

tests_folder = Path(__file__).absolute().parent.parent


def mock_run(command):
    print(command)


def mock_for_tests(*, p_open=None, subprocess=None):
    if subprocess is not None:
        subprocess_instance = mock.MagicMock()
        subprocess_instance.run.side_effect = mock_run
        subprocess.return_value = subprocess_instance
    if p_open is not None:
        p_open_instance = mock.MagicMock()
        p_open_instance.communicate.return_value = (
            b"Mocked communication output",
            b"Mocked communication error"
        )
        p_open.return_value = p_open_instance


@mock.patch("auto_sbatch.processes.subprocess")
@mock.patch("auto_sbatch.sbatch.Popen")
def test_launch(p_open_mock, subprocess_mock, capsys):
    mock_for_tests(p_open=p_open_mock, subprocess=subprocess_mock)

    args = get_args(
        use_local=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml",
        ],
        verbose=False,
        use_schema=False,
    )
    slurm_command = "python {script_name} {all_params}"
    cli_args = OmegaConf.from_dotlist(
        [
            "slurm.run_work_directory='tests/slurm'",
            "slurm.script='train'",
            f"slurm.command='{slurm_command}'",
        ]
    )
    launch(args, cli_args)

    captured_out = capsys.readouterr().out.strip("\n")
    slurm_script = SlurmScriptParser(captured_out, slurm_command)
    slurm_script.parse()
    assert slurm_script.script_name == "train.py"
    assert "slurm" in slurm_script.params.keys()
    assert "script" in slurm_script.params.slurm.keys()


@mock.patch("auto_sbatch.processes.subprocess")
@mock.patch("auto_sbatch.sbatch.Popen")
def test_launch_grid_search(p_open_mock, subprocess_mock, capsys):
    mock_for_tests(p_open=p_open_mock, subprocess=subprocess_mock)

    args = get_args(
        use_local=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml",
            tests_folder / "configs/test_slurm_launch_grid_search.yaml",
        ],
        verbose=False,
        use_schema=False,
    )
    slurm_command = "python {script_name} {all_params}"
    cli_args = OmegaConf.from_dotlist(
        [
            "slurm.run_work_directory='tests/slurm'",
            "slurm.script='train'",
            f"slurm.command='{slurm_command}'",
            "slurm.grid_search=['seed']"
        ]
    )
    launch(args, cli_args)

    captured_out = capsys.readouterr().out.strip("\n")
    slurm_script = SlurmScriptParser(captured_out, slurm_command)
    slurm_script.parse()
    assert slurm_script.script_name == "train.py"
    assert "slurm" in slurm_script.params.keys()
    assert "script" in slurm_script.params.slurm.keys()
    assert "seed" in slurm_script.params.keys()


def test_grid_search_exclusion_from_past_search():
    result = grid_search_exclusion_from_past_search(
        ["seed=[0.]", "losses.coefs.contrastive=[0.1,0.2]"]
    )
    assert result == [
        {"seed": 0., "losses.coefs.contrastive": 0.1},
        {"seed": 0., "losses.coefs.contrastive": 0.2},
    ]
