import logging
from collections import defaultdict
from functools import cached_property
from pathlib import Path

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from icenet_mp.types import ArrayTCHW, DataloaderArgs, DataSpace, Hemisphere, MaskType
from icenet_mp.utils import mask_dir

from .combined_dataset import CombinedDataset
from .single_dataset import SingleDataset

logger = logging.getLogger(__name__)


class CommonDataModule(LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        """Initialise a CommonDataModule from a config.

        The config specifies all datasets used and how to group them. Data splits are
        also determined from the config, and the appropriate data loaders are created.
        """
        super().__init__()

        # Load paths
        self.base_path = Path(config["base_path"])

        # Construct dataset groups
        self.dataset_groups = defaultdict(list)
        self.dataset_variables: dict[str, set[str]] = defaultdict(set)
        for dataset in config["data"]["datasets"].values():
            self.dataset_groups[dataset["group_as"]].append(
                (
                    self.base_path / "data" / "anemoi" / f"{dataset['name']}.zarr"
                ).resolve()
            )
            if variables := dataset.get("variables"):
                self.dataset_variables[dataset["group_as"]].update(variables)
        logger.info("Found %d dataset groups.", len(self.dataset_groups))
        for idx, (name, paths) in enumerate(self.dataset_groups.items(), start=1):
            logger.info("%d) %s:", idx, name)
            for path in paths:
                logger.info("%s - %s", " " * (len(str(idx)) + 1), path)
            if name in self.dataset_variables:
                logger.info(
                    "%s selecting variables: %s",
                    " " * (len(str(idx)) + 1),
                    sorted(self.dataset_variables[name]),
                )

        # Check prediction target
        self.target_group_name = config["predict"]["target"]["group_name"]
        if self.target_group_name not in self.dataset_groups:
            msg = f"Could not find prediction target {self.target_group_name}."
            raise ValueError(msg)
        self._target_variables: list[str] = config["predict"]["target"].get(
            "variables", []
        )

        # Set periods for train, validation, and test
        self.batch_size = int(config["data"]["split"]["batch_size"])
        # Optional stride over training window start dates (see CombinedDataset):
        # consecutive windows overlap in all but one timestep, so a stride of e.g. 2
        # halves the samples per epoch while barely reducing data coverage. Only
        # applied to training; validation/test/predict always use every window.
        self.train_stride = int(config["data"]["split"].get("train_stride", 1))
        self.predict_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["predict"]
        ]
        self.test_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["test"]
        ]
        self.train_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["train"]
        ]
        self.val_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["validate"]
        ]

        # Set history and forecast steps
        self.n_forecast_steps = int(config["predict"].get("n_forecast_steps", 1))
        self.n_history_steps = int(config["predict"].get("n_history_steps", 1))

        # Set common arguments for the dataloader
        self._common_dataloader_kwargs = DataloaderArgs(
            batch_sampler=None,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,  # must be None when num_workers=0
            sampler=None,
            worker_init_fn=None,
        )

    @cached_property
    def datasets(self) -> dict[str, SingleDataset]:
        """Return a dictionary of dataset group names to SingleDataset objects."""
        return {
            name: SingleDataset(
                name,
                paths,
                variables=sorted(self.dataset_variables.get(name, ())),
            )
            for name, paths in self.dataset_groups.items()
        }

    @cached_property
    def hemisphere(self) -> Hemisphere:
        """Return the hemisphere of the dataset."""
        hemisphere: set[Hemisphere] = {ds.hemisphere for ds in self.datasets.values()}
        if len(hemisphere) != 1:
            msg = f"Found {len(hemisphere)} different hemisphere indicators across {len(self.dataset_groups)} dataset groups."
            raise ValueError(msg)
        return hemisphere.pop()

    @cached_property
    def input_spaces(self) -> list[DataSpace]:
        """Return the data space for each input."""
        return [ds.space for ds in self.datasets.values()]

    @cached_property
    def latitudes(self) -> dict[str, list[float]]:
        """Return the latitudes of the dataset."""
        return {name: ds.latitudes for name, ds in self.datasets.items()}

    @cached_property
    def longitudes(self) -> dict[str, list[float]]:
        """Return the longitudes of the dataset."""
        return {name: ds.longitudes for name, ds in self.datasets.items()}

    @cached_property
    def mask_directory(self) -> Path:
        """Mask directory for the prediction target group.

        A target group usually holds a single dataset with generated masks, but if it
        holds several, pick the first. Combining masks across datasets is unsupported.
        """
        paths = self.dataset_groups[self.target_group_name]
        available = [
            path
            for path in paths
            if any(
                (mask_dir(self.base_path, path.stem) / f"{kind}_mask.npy").exists()
                for kind in (MaskType.ACTIVE, MaskType.LAND)
            )
        ]
        chosen = (available or paths)[0].stem
        if len(paths) > 1:
            logger.warning(
                "Target group %r has %d datasets; using %r for masks "
                "(combining masks across datasets is not supported).",
                self.target_group_name,
                len(paths),
                chosen,
            )
        return mask_dir(self.base_path, chosen)

    @cached_property
    def output_space(self) -> DataSpace:
        """Return the data space of the desired output."""
        return (
            self.datasets[self.target_group_name]
            .subset(variables=self.target_variables)
            .space
        )

    @cached_property
    def target_variables(self) -> list[str]:
        """Return the names of the variables to predict."""
        if self._target_variables:
            return self._target_variables
        return self.variable_names[self.target_group_name]

    @cached_property
    def target_variable_indices(self) -> list[int]:
        """Return the indices of the variables to predict."""
        return [
            self.variable_names[self.target_group_name].index(variable)
            for variable in self.target_variables
        ]

    @cached_property
    def variable_names(self) -> dict[str, list[str]]:
        """Return the variable names for each input."""
        return {ds.name: ds.variable_names for ds in self.datasets.values()}

    def assign_workers(self, n_workers: int) -> None:
        """Assign number of workers for data loading."""
        logger.info("Assigning %d workers for data loading.", n_workers)
        self._common_dataloader_kwargs["num_workers"] = n_workers
        self._common_dataloader_kwargs["persistent_workers"] = n_workers > 0
        self._common_dataloader_kwargs["prefetch_factor"] = 1 if n_workers > 0 else None

    def predict_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct predict dataloader."""
        dataset = CombinedDataset(
            [
                ds.subset(date_ranges=self.predict_periods)
                for ds in self.datasets.values()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded predict dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct test dataloader."""
        dataset = CombinedDataset(
            [ds.subset(date_ranges=self.test_periods) for ds in self.datasets.values()],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded test dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    def train_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct train dataloader."""
        dataset = CombinedDataset(
            [
                ds.subset(date_ranges=self.train_periods)
                for ds in self.datasets.values()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            stride=self.train_stride,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded training dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct validation dataloader."""
        dataset = CombinedDataset(
            [ds.subset(date_ranges=self.val_periods) for ds in self.datasets.values()],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded validation dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
