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

log = logging.getLogger(__name__)


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
        log.info("Found %d dataset groups.", len(self.dataset_groups))
        for idx, (name, paths) in enumerate(self.dataset_groups.items(), start=1):
            log.info("%d) %s:", idx, name)
            for path in paths:
                log.info("%s - %s", " " * (len(str(idx)) + 1), path)

        # Requested input variables
        self._requested_input_variables: dict[str, list[str]] = {
            str(group_name): [str(v) for v in variable_names]
            for group_name, variable_names in config["variables"]["input"].items()
        }

        # Requested target variables
        self._requested_target_variables: dict[str, list[str]] = {
            str(group_name): [str(v) for v in variable_names]
            for group_name, variable_names in config["variables"]["target"].items()
        }

        # Set periods for train, validation, and test
        self.batch_size = int(config["window"]["batch_size"])
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
        self.n_forecast_steps = int(config["window"].get("n_forecast_steps", 1))
        self.n_history_steps = int(config["window"].get("n_history_steps", 1))

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
        """Return a filtered dictionary of dataset group names to SingleDataset objects.

        Only include requested variables for each dataset group. If no variables are
        requested for a dataset group, ignore it.
        """
        return {
            name: self.datasets_unfiltered[name].subset(variables=variables)
            for name, variables in self.variable_names.items()
        }

    @cached_property
    def datasets_unfiltered(self) -> dict[str, SingleDataset]:
        """Return an unfiltered dictionary of dataset group names to SingleDataset objects."""
        return {
            name: SingleDataset(name, paths)
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
            log.warning(
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
    def target_group_name(self) -> str:
        """Return the name of the target variable group."""
        target_variable_groups = list(self._requested_target_variables.keys())
        if len(target_variable_groups) != 1:
            msg = (
                f"Expected exactly one target variable group, but found "
                f"{len(target_variable_groups)}: {target_variable_groups}."
            )
            raise ValueError(msg)
        if target_variable_groups[0] not in self.dataset_groups:
            available_ds_groups = ", ".join(sorted(self.dataset_groups)) or "<none>"
            msg = (
                f"Target dataset group {target_variable_groups[0]!r} was not found in "
                f"list of dataset groups. Available groups: {available_ds_groups}."
            )
            raise ValueError(msg)
        return target_variable_groups[0]

    @cached_property
    def target_variables(self) -> list[str]:
        """Return the names of the variables to predict."""
        available_variables = next(
            ds.variable_names
            for ds in self.datasets.values()
            if ds.name == self.target_group_name
        )
        requested_variables = self._requested_target_variables[self.target_group_name]
        for requested_variable in requested_variables:
            if requested_variable not in available_variables:
                available_ = ", ".join(sorted(available_variables)) or "<none>"
                msg = (
                    f"Target variable {requested_variable!r} was not found in dataset "
                    f"group {self.target_group_name!r}. Available variables: "
                    f"{available_}."
                )
                raise ValueError(msg)
        return requested_variables

    @cached_property
    def target_variable_indices(self) -> list[int]:
        """Return the indices of the target variables within their dataset group."""
        try:
            return [
                self.variable_names[self.target_group_name].index(variable)
                for variable in self.target_variables
            ]
        except ValueError as exc:
            msg = (
                f"Not all target variable {self.target_variables} were found in the "
                f"dataset group {self.target_group_name!r}. Available variables: "
                f"{self.variable_names[self.target_group_name]!r}."
            )
            raise ValueError(msg) from exc

    @cached_property
    def variable_names(self) -> dict[str, list[str]]:
        """Return the variable names for each input dataset group."""
        available = {
            ds.name: ds.variable_names for ds in self.datasets_unfiltered.values()
        }
        if not self._requested_input_variables:
            return available
        verified: dict[str, list[str]] = {}
        for group_name, variable_names in self._requested_input_variables.items():
            if group_name not in self.dataset_groups:
                available_ = ", ".join(sorted(self.dataset_groups)) or "<none>"
                msg = (
                    f"Input variable group {group_name!r} was not found in the "
                    f"configured datasets. Available groups: {available_}."
                )
                raise ValueError(msg)
            verified[group_name] = []
            for variable in variable_names:
                if variable not in available[group_name]:
                    available_variables = (
                        ", ".join(sorted(available[group_name])) or "<none>"
                    )
                    msg = (
                        f"Input variable {variable!r} was not found in dataset group "
                        f"{group_name!r}. Available variables: {available_variables}."
                    )
                    raise ValueError(msg)
                verified[group_name].append(variable)
        return verified

    def assign_workers(self, n_workers: int) -> None:
        """Assign number of workers for data loading."""
        log.info("Assigning %d workers for data loading.", n_workers)
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
        for line in dataset.variable_list():
            log.info(line)
        log.info(
            "Loaded predict dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date.astype("datetime64[m]"),
            dataset.end_date.astype("datetime64[m]"),
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
        for line in dataset.variable_list():
            log.info(line)
        log.info(
            "Loaded test dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date.astype("datetime64[m]"),
            dataset.end_date.astype("datetime64[m]"),
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
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
        )
        for line in dataset.variable_list():
            log.info(line)
        log.info(
            "Loaded training dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date.astype("datetime64[m]"),
            dataset.end_date.astype("datetime64[m]"),
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
        log.info(
            "Loaded validation dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date.astype("datetime64[m]"),
            dataset.end_date.astype("datetime64[m]"),
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
