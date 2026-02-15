# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#     "numpy>=1.24.0",
# ]
# ///

import json
from argparse import ArgumentParser
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
import requests
from temporaldata import ArrayDict, RegularTimeSeries

from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.format import NeuralData
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech
from brainsets.taxonomy.subject import Species
from brainsets.utils.zip_utils import download_and_extract

parser = ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Whether to overwrite already processed data. Use with caution, as this will re-process all data and overwrite existing processed files.",
)
parser.add_argument(
    "--allow_corrupted", action="store_true", help="Allow processing of corrupted data."
)
parser.add_argument(
    "--skip_initial_download",
    action="store_true",
    help="Skip the initial download of raw data. Use with caution, only if you are sure the raw data has already been downloaded and is in the correct format.",
)

logger = getLogger(__name__)

SAMPLING_RATE = 2048  # Hz
ALL_SUBJECT_TRIALS = [
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 0),
    (3, 1),
    (3, 2),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 0),
    (6, 0),
    (6, 1),
    (6, 4),
    (7, 0),
    (7, 1),
    (8, 0),
    (9, 0),
    (10, 0),
    (10, 1),
]


def _to_subject_id(subj_id: int | str) -> str:
    subj_id = str(subj_id)
    return subj_id if subj_id.startswith("sub_") else f"sub_{subj_id}"


def corrupted_electrodes_path(root: Path) -> Path:
    return root / "corrupted_elec.json"


def electrode_labels_path(raw_dir: Path, subject_id: int | str) -> Path:
    return (
        raw_dir
        / "electrode_labels"
        / _to_subject_id(subject_id)
        / "electrode_labels.json"
    )


def localization_path(raw_dir: Path, subject_id: int | str) -> Path:
    return raw_dir / "localization" / _to_subject_id(subject_id) / "depth-wm.csv"


class Pipeline(BrainsetPipeline):
    brainset_description = BrainsetDescription(
        id="wang_barbu_braintreebank_2023",
        origin_version="0.1.0",
        derived_version="0.1.0",
        source="https://braintreebank.dev/",
        description="""
        Large-scale dataset of electrophysiological neural responses, recorded from 
        intracranial probes while 10 subjects watched one or more Hollywood movies.""",
    )

    device_description = DeviceDescription(
        id="intracranial_probes",
        recording_tech=RecordingTech.STEREO_EEG,
    )

    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        assert args is not None

        if not args.skip_initial_download:
            cls._initial_download(raw_dir)

        manifest = (
            pd.DataFrame(
                {
                    "subject_id": _to_subject_id(subj_id),
                    "session_id": f"trial{trial_id:03}",
                }
                for subj_id, trial_id in ALL_SUBJECT_TRIALS
            )
            .assign(neural_data=lambda df: df["subject_id"] + "_" + df["session_id"])
            .set_index("neural_data")
        )

        return manifest

    def download(self, manifest_item: NamedTuple) -> Path:
        self.update_status("Downloading...")
        path = self.processed_dir / manifest_item.neural_data
        url = f"https://braintreebank.dev/data/subject_data/{manifest_item.subject_id}/{manifest_item.session_id}/{manifest_item.neural_data}.h5.zip"
        download_and_extract(url, extract_to=path.parent)
        return path

    def process(self, manifest_item: NamedTuple, downloaded_path: Path) -> None:
        self.update_status("Processing...")
        subject_id = manifest_item.subject_id
        session_id = manifest_item.session_id
        assert self.args is not None

        # Check if already processed
        output_path = self.processed_dir / f"{manifest_item.neural_data}.hdf5"
        if output_path.exists() and not self.args.overwrite:
            self.update_status("Already processed, skipping.")
            logger.info("Skipping processing, file exists: %s", output_path)
            return

        self.update_status("Loading electrode metadata...")
        with open(electrode_labels_path(self.raw_dir, subject_id)) as f:
            electrode_labels = [self._clean_electrode_label(e) for e in json.load(f)]
        channels = self._load_ieeg_electrodes(subject_id, electrode_labels)

        self.update_status("Loading neural data...")
        neural_data = self._load_ieeg_data(downloaded_path, electrode_labels, channels)

        self.update_status("Saving processed data...")

        subject = SubjectDescription(
            id=subject_id,
            species=Species.HUMAN,
        )

        session = SessionDescription(id=session_id, recording_date=datetime.min)

        brainset = NeuralData(
            brainset=self.brainset_description,
            subject=subject,
            session=session,
            device=self.device_description,
            data=neural_data,
            channels=channels,
        )

        with h5py.File(output_path, "w") as f:
            brainset.to_hdf5(f)
        logger.info("Processed data saved to %s", output_path)

        # Delete the downloaded raw data to save space
        downloaded_path.unlink()

        self.update_status("Processing complete.")
        n_electrodes = neural_data.data.shape[1]
        session_length = neural_data.data.shape[0] / neural_data.sampling_rate

        logger.info(
            "Processed session %s for subject %s at %s:\n"
            "\t- Number of electrodes: %d\n"
            "\t- Session length: %.2f seconds\n",
            session_id,
            subject_id,
            output_path,
            n_electrodes,
            session_length,
        )

    @classmethod
    def _initial_download(cls, raw_dir: Path):
        """Called when the pipeline is first run, to download the raw data."""
        all_subjects = {subj for subj, _ in ALL_SUBJECT_TRIALS}

        # Download corrupted electrodes metadata (if not already downloaded)
        path = corrupted_electrodes_path(raw_dir)
        if not path.exists():
            logger.info("Downloading corrupted electrodes metadata to %s...", path)
            url = "https://braintreebank.dev/data/corrupted_elec.json"
            response = requests.get(url)
            response.raise_for_status()
            corrupted_electrodes = response.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(corrupted_electrodes))
        else:
            logger.info(
                "Corrupted electrodes metadata already exists at %s, skipping download.",
                path,
            )

        # Download electrode labels
        if all(electrode_labels_path(raw_dir, subj).exists() for subj in all_subjects):
            logger.info(
                "Electrode labels already exist for all subjects, skipping download."
            )
        else:
            logger.info("Downloading electrode labels...")
            download_and_extract(
                "https://braintreebank.dev/data/electrode_labels.zip",
                extract_to=raw_dir,
                chunk_size=None,  # No streaming necessary
            )
            logger.info("Electrode labels downloaded and extracted to %s.", raw_dir)

        # Download localization data
        if all(localization_path(raw_dir, subj).exists() for subj in all_subjects):
            logger.info(
                "Localization data already exists for all subjects, skipping download."
            )
        else:
            logger.info("Downloading localization data...")
            download_and_extract(
                "https://braintreebank.dev/data/localization.zip",
                extract_to=raw_dir,
                chunk_size=None,  # No streaming necessary
            )
            logger.info("Localization data downloaded and extracted to %s.", raw_dir)

    def _load_ieeg_electrodes(
        self, subject_id: str, electrode_labels: list[str]
    ) -> ArrayDict:
        """Load and clean electrode channel metadata."""

        electrode_labels = self._filter_electrode_labels(subject_id, electrode_labels)

        # Load and clean localization data
        df = pd.read_csv(localization_path(self.raw_dir, subject_id))
        df["Electrode"] = df["Electrode"].map(self._clean_electrode_label)
        df = df.set_index("Electrode")

        # Select relevant electrodes
        df = df.loc[electrode_labels]

        # Convert LPI → MNI (RAS) by flipping signs
        # NOTE: this is not the same as the MNI space used in the BIDS specification.
        # Awaiting proper MNI coordinates from braintreebank.
        coordinates = -df[["L", "P", "I"]].to_numpy(dtype=np.float32)

        return ArrayDict(
            id=np.array(electrode_labels),
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
        )

    def _load_ieeg_data(
        self, neural_data_file: Path, electrode_labels: list[str], channels: ArrayDict
    ) -> RegularTimeSeries:
        """Load the neural data from the provided h5 file,
        using the electrode labels to select and order the channels."""
        with h5py.File(neural_data_file, "r", locking=False) as f:
            data_group = f["data"]
            labels = list(channels.id)  # type: ignore[attr-defined]

            # Build reverse index
            label_to_index = {label: i for i, label in enumerate(electrode_labels)}

            first_idx = label_to_index[labels[0]]
            data_len = data_group[f"electrode_{first_idx}"].shape[0]

            neural_data = np.empty((data_len, len(labels)), dtype=np.float32)

            for i, label in enumerate(labels):
                idx = label_to_index[label]
                neural_data[:, i] = data_group[f"electrode_{idx}"]

        return RegularTimeSeries(
            data=neural_data,
            sampling_rate=SAMPLING_RATE,
            domain_start=0.0,
            domain="auto",
        )

    def _filter_electrode_labels(
        self, subject_id: str, electrode_labels: list[str]
    ) -> list[str]:
        """
        Filter the electrode labels to remove corrupted electrodes
        and electrodes that don't have brain signal
        """
        assert self.args is not None

        corrupted: set[str] = set()
        if not self.args.allow_corrupted:
            with open(corrupted_electrodes_path(self.raw_dir)) as f:
                corrupted = {
                    self._clean_electrode_label(e) for e in json.load(f)[subject_id]
                }
        return [
            e
            for e in electrode_labels
            if e not in corrupted and not e.upper().startswith(("DC", "TRIG"))
        ]

    def _clean_electrode_label(self, label: str) -> str:
        """Remove special characters from the electrode label (e.g. "*", "#")
        to match the electrode labels in the h5 files."""
        return label.replace("*", "").replace("#", "")
