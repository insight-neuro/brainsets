from collections.abc import Callable
from typing import Literal

from temporaldata import ArrayDict, Data, Interval, RegularTimeSeries

from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)

from .core import serialize_fn_map


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
        super().__init__(*args, **kwargs, domain=domain)
        self.brainset = brainset
        self.subject = subject
        self.session = session
        self.device = device
        self.data = data
        self.channels = channels

    def to_hdf5(
        self, file, serialize_fn_map: dict[type, Callable] | None = serialize_fn_map
    ):
        return super().to_hdf5(file, serialize_fn_map)
