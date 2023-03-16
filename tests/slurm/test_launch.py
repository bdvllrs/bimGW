from pathlib import Path
from unittest import mock

from auto_sbatch import SlurmScriptParser
from omegaconf import OmegaConf

from bim_gw.utils import get_args
from bim_gw.utils.config import get_argv_dotlist
from slurm.launch import main


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
            Path("../configs/test_base.yaml")
        ],
        verbose=False,
        use_schema=False,
    )
    slurm_command = "python {script_name} {all_params}"
    cli_args = OmegaConf.from_dotlist(
        get_argv_dotlist(
            [
                "slurm.run_work_directory='tests/slurm'",
                "slurm.script='train'",
                f"slurm.command='{slurm_command}'",
            ]
        )
    )
    main(args, cli_args)

    captured_out = capsys.readouterr().out.strip("\n")
    slurm_script = SlurmScriptParser(captured_out, slurm_command)
    slurm_script.parse()
    assert slurm_script.run_script == "train.py"
    assert "slurm" in slurm_script.params.keys()
    assert "script" in slurm_script.params.slurm.keys()


@mock.patch("auto_sbatch.processes.subprocess")
@mock.patch("auto_sbatch.sbatch.Popen")
def test_launch_grid_search(p_open_mock, subprocess_mock, capsys):
    mock_for_tests(p_open=p_open_mock, subprocess=subprocess_mock)

    args = get_args(
        use_local=False,
        additional_config_files=[
            Path("../configs/test_base.yaml"),
            Path("../configs/test_slurm_launch_grid_search.yaml"),
        ],
        verbose=False,
        use_schema=False,
    )
    slurm_command = "python {script_name} {all_params}"
    cli_args = OmegaConf.from_dotlist(
        get_argv_dotlist(
            [
                "slurm.run_work_directory='tests/slurm'",
                "slurm.script='train'",
                f"slurm.command='{slurm_command}'",
                "slurm.grid_search=['seed']"
            ]
        )
    )
    main(args, cli_args)

    captured_out = capsys.readouterr().out.strip("\n")
    slurm_script = SlurmScriptParser(captured_out, slurm_command)
    slurm_script.parse()
    assert slurm_script.run_script == "train.py"
    assert "slurm" in slurm_script.params.keys()
    assert "script" in slurm_script.params.slurm.keys()
    assert "seed" in slurm_script.params.keys()
