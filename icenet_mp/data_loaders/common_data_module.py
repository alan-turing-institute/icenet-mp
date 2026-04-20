import logging
from collections import defaultdict
from functools import cached_property
from pathlib import Path

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from icenet_mp.types import ArrayTCHW, DataloaderArgs, DataSpace, Hemisphere

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
        for dataset in config["data"]["datasets"].values():
            self.dataset_groups[dataset["group_as"]].append(
                (
                    self.base_path / "data" / "anemoi" / f"{dataset['name']}.zarr"
                ).resolve()
            )
        logger.info("Found %d dataset groups.", len(self.dataset_groups))
        for idx, (name, paths) in enumerate(self.dataset_groups.items(), start=1):
            logger.info("%d) %s:", idx, name)
            for path in paths:
                logger.info("%s - %s", " " * (len(str(idx)) + 1), path)

        # Check prediction target
        self.target_group_name = config["predict"]["target"]["group_name"]
        if self.target_group_name not in self.dataset_groups:
            msg = f"Could not find prediction target {self.target_group_name}."
            raise ValueError(msg)
        self.target_variables: list[str] = config["predict"]["target"].get(
            "variables", []
        )

        # Set periods for train, validation, and test
        self.batch_size = int(config["data"]["split"]["batch_size"])
        self.predict_periods = [
            {k: None if v == "None" else v for k, v in period.items()}
            for period in config["data"]["split"]["predict"]
        ]
        self.test_periods = [
            {k: None if v == "None" else v for k, v in period.items()}
            for period in config["data"]["split"]["test"]
        ]
        self.train_periods = [
            {k: None if v == "None" else v for k, v in period.items()}
            for period in config["data"]["split"]["train"]
        ]
        self.val_periods = [
            {k: None if v == "None" else v for k, v in period.items()}
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
            sampler=None,
            worker_init_fn=None,
        )

    @property
    def hemisphere(self) -> Hemisphere:
        """Return the hemisphere of the dataset."""
        hemisphere: set[Hemisphere] = {
            SingleDataset(name, paths).hemisphere
            for name, paths in self.dataset_groups.items()
        }
        if len(hemisphere) != 1:
            msg = f"Found {len(hemisphere)} different hemisphere indicators across {len(self.dataset_groups)} dataset groups."
            raise ValueError(msg)
        return hemisphere.pop()

    @cached_property
    def input_spaces(self) -> list[DataSpace]:
        """Return the data space for each input."""
        return [
            SingleDataset(name, paths).space
            for name, paths in self.dataset_groups.items()
        ]

    @property
    def latitudes(self) -> dict[str, list[float]]:
        """Return the latitudes of the dataset."""
        return {
            name: SingleDataset(name, paths).latitudes
            for name, paths in self.dataset_groups.items()
        }

    @property
    def longitudes(self) -> dict[str, list[float]]:
        """Return the longitudes of the dataset."""
        return {
            name: SingleDataset(name, paths).longitudes
            for name, paths in self.dataset_groups.items()
        }

    @cached_property
    def output_space(self) -> DataSpace:
        """Return the data space of the desired output."""
        return next(
            SingleDataset(name, paths, variables=self.target_variables).space
            for name, paths in self.dataset_groups.items()
            if name == self.target_group_name
        )

    def assign_workers(self, n_workers: int) -> None:
        """Assign number of workers for data loading."""
        logger.info("Assigning %d workers for data loading.", n_workers)
        self._common_dataloader_kwargs["num_workers"] = n_workers
        self._common_dataloader_kwargs["persistent_workers"] = n_workers > 0

    def predict_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct predict dataloader."""
        dataset = CombinedDataset(
            [
                SingleDataset(
                    name,
                    paths,
                    date_ranges=self.predict_periods,
                )
                for name, paths in self.dataset_groups.items()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded predict dataset with %d samples between %s and %s.",
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
            [
                SingleDataset(
                    name,
                    paths,
                    date_ranges=self.test_periods,
                )
                for name, paths in self.dataset_groups.items()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded test dataset with %d samples between %s and %s.",
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
                SingleDataset(
                    name,
                    paths,
                    date_ranges=self.train_periods,
                )
                for name, paths in self.dataset_groups.items()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded training dataset with %d samples between %s and %s.",
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
            [
                SingleDataset(
                    name,
                    paths,
                    date_ranges=self.val_periods,
                )
                for name, paths in self.dataset_groups.items()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        logger.info(
            "Loaded validation dataset with %d samples between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
