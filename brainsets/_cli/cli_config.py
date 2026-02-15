from pathlib import Path

import click
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.shortcuts import CompleteStyle

from .utils import expand_path, load_config, save_config


@click.command()
@click.option(
    "--raw-dir",
    help="Path for storing raw data.",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
@click.option(
    "--processed-dir",
    help="Path for storing processed brainsets.",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
def config(raw_dir: Path | None, processed_dir: Path | None):
    """Set raw and processed data directories."""

    # Get missing args from user prompts
    if raw_dir is None or processed_dir is None:
        raw_dir = prompt(
            "Enter raw data directory: ",
            completer=PathCompleter(only_directories=True),
            complete_style=CompleteStyle.READLINE_LIKE,
        )
        processed_dir = prompt(
            "Enter processed data directory: ",
            completer=PathCompleter(only_directories=True),
            complete_style=CompleteStyle.READLINE_LIKE,
        )

    raw_dir = expand_path(raw_dir)
    processed_dir = expand_path(processed_dir)

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = load_config(raise_cli_error=False)
    config_exists = config is not None
    if not config_exists:
        config = {}
    config["raw_dir"] = str(raw_dir)
    config["processed_dir"] = str(processed_dir)
    config_filepath = save_config(config)

    if not config_exists:
        click.echo(f"Created config file at {config_filepath}")
    else:
        click.echo(f"Updated config file at {config_filepath}")

    click.echo(f"Raw data dir: {config['raw_dir']}")
    click.echo(f"Processed data dir: {config['processed_dir']}")
