# How-to guides

Step-by-step guides for common tasks.

- [Add a model](add-a-model.md) - implement a custom architecture
- [Add a processor](add-a-processor.md) - implement a processor, including models with different training and inference behaviour
- [Train a model](train.md) - run single-stage end-to-end training
- [Train in stages](train-multistage.md) - pretrain each component separately before finetuning
- [Evaluate a model](evaluate.md) - evaluate a trained model
- [Run VIF analysis for multicollinearity](vif-analysis.md) - identify redundant input variables before training
- [Run feature screening](feature-screening.md) - PCA, EOF, Random Forest importance, correlation, and the consolidated evidence report
- [Feature screening demonstration: full year 2020 run](feature-screening-year-demo.md) - side-by-side `sic_change` vs `absolute` run with plots and tables, demonstrating the new additions

For developers, there is also a guide on the process to follow to [create a new release](release.md).
