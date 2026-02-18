from collections.abc import Callable
from typing import Literal

from h5py import File
from temporaldata import ArrayDict, Data, Interval, RegularTimeSeries

from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)

from .core import serialize_fn_map


def bids_filename(
    subject_id: str | int,
    session_id: str | int,
    *suffixes: str,
    extension: str = "h5",
) -> str:
    """Generate a standardized BIDS filename."""

    # Normalize to string and remove accidental prefixes
    subject_id = str(subject_id).replace("sub-", "")
    session_id = str(session_id).replace("ses-", "")

    # Zero-pad
    subject_id = subject_id.zfill(3)
    session_id = session_id.zfill(2)

    suffix_str = f"_{'_'.join(suffixes)}" if suffixes else ""

    return f"sub-{subject_id}_ses-{session_id}{suffix_str}.{extension}"


class NeuralData(Data):
    """Standardized format for neural data."""

    brainset: BrainsetDescription
    """Brainset-level metadata describing the dataset as a whole."""
    subject: SubjectDescription
    """Subject-level metadata describing the individual subject."""
    session: SessionDescription
    """Session-level metadata describing the recording session."""
    device: DeviceDescription
    """Device-level metadata describing the recording device."""
    data: RegularTimeSeries
    """Time series data containing the neural recordings."""
    channels: ArrayDict
    """ArrayDict containing metadata for each channel in the neural data."""

    def __init__(
        self,
        brainset: BrainsetDescription,
        subject: SubjectDescription,
        session: SessionDescription,
        device: DeviceDescription,
        data: RegularTimeSeries,
        channels: ArrayDict,
        domain: Literal["auto"] | Interval = "auto",
        *args,
        **kwargs,
    ):
        self.brainset = brainset
        self.subject = subject
        self.session = session
        self.device = device

        super().__init__(*args, **kwargs, data=data, channels=channels, domain=domain)

    def to_hdf5(
        self,
        file: File,
        serialize_fn_map: dict[type, Callable] | None = serialize_fn_map,
    ):
        super().to_hdf5(file, serialize_fn_map)
