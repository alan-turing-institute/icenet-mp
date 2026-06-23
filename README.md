# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/icenet-mp/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                            |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| icenet\_mp/\_\_init\_\_.py                                      |        9 |        0 |    100% |           |
| icenet\_mp/callbacks/\_\_init\_\_.py                            |        6 |        0 |    100% |           |
| icenet\_mp/callbacks/activation\_saver.py                       |       92 |       70 |     24% |63-73, 83-116, 124-127, 135, 143, 146-162, 173-180, 192-218, 231-245, 253-258 |
| icenet\_mp/callbacks/ema\_weight\_averaging\_callback.py        |       13 |        8 |     38% |23-27, 33-39 |
| icenet\_mp/callbacks/metric\_summary\_callback.py               |       66 |       22 |     67% |49-52, 82-83, 90, 98-99, 103-106, 114-115, 121-126, 135-142, 148 |
| icenet\_mp/callbacks/plotting\_callback.py                      |      107 |       71 |     34% |70-73, 79-94, 104-138, 151-166, 171-180, 193-212, 219-228 |
| icenet\_mp/callbacks/unconditional\_checkpoint.py               |       21 |       10 |     52% |17-19, 24, 29-30, 34-35, 39-40 |
| icenet\_mp/cli/\_\_init\_\_.py                                  |        2 |        0 |    100% |           |
| icenet\_mp/cli/datasets.py                                      |       38 |       20 |     47% |27-34, 47-53, 66-69, 73 |
| icenet\_mp/cli/evaluate.py                                      |       18 |        5 |     72% | 39-44, 48 |
| icenet\_mp/cli/hydra.py                                         |       29 |        0 |    100% |           |
| icenet\_mp/cli/main.py                                          |       23 |        8 |     65% | 39-51, 55 |
| icenet\_mp/cli/train.py                                         |       14 |        3 |     79% | 20-21, 25 |
| icenet\_mp/compatibility/\_\_init\_\_.py                        |       17 |        0 |    100% |           |
| icenet\_mp/compatibility/lightning/\_\_init\_\_.py              |        9 |        0 |    100% |           |
| icenet\_mp/compatibility/lightning/xpu\_accelerator.py          |       72 |       39 |     46% |35-42, 55, 60-62, 77-111, 125-127, 133, 144-147, 152 |
| icenet\_mp/compatibility/torch/\_\_init\_\_.py                  |        3 |        0 |    100% |           |
| icenet\_mp/compatibility/torch/patch\_interpolate\_antialias.py |       11 |        8 |     27% |     14-26 |
| icenet\_mp/compatibility/torch/patch\_parameter\_deepcopy.py    |       12 |        7 |     42% |     16-28 |
| icenet\_mp/config/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| icenet\_mp/data\_loaders/\_\_init\_\_.py                        |        4 |        0 |    100% |           |
| icenet\_mp/data\_loaders/combined\_dataset.py                   |       42 |        1 |     98% |        88 |
| icenet\_mp/data\_loaders/common\_data\_module.py                |       79 |       27 |     66% |48-49, 93, 101-105, 110, 115, 120, 125, 133-135, 141-157, 163-176, 182-198, 204-217 |
| icenet\_mp/data\_loaders/single\_dataset.py                     |      141 |        5 |     96% |179, 184, 208, 282-287 |
| icenet\_mp/data\_processors/\_\_init\_\_.py                     |        2 |        0 |    100% |           |
| icenet\_mp/data\_processors/data\_downloader.py                 |      158 |       96 |     39% |56-57, 94-134, 138-155, 178-235, 240-243, 252-254, 262-296 |
| icenet\_mp/data\_processors/data\_downloader\_factory.py        |       11 |        4 |     64% |     19-26 |
| icenet\_mp/data\_processors/filters/\_\_init\_\_.py             |       13 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/nan\_to\_num\_filter.py     |        9 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/reproject\_filter.py        |       31 |        1 |     97% |       111 |
| icenet\_mp/data\_processors/filters/set\_geography\_filter.py   |       19 |        1 |     95% |        51 |
| icenet\_mp/data\_processors/preprocessors/\_\_init\_\_.py       |        4 |        0 |    100% |           |
| icenet\_mp/data\_processors/preprocessors/icenet\_sic.py        |       64 |       46 |     28% |21-27, 31, 36-67, 75-132 |
| icenet\_mp/data\_processors/preprocessors/ipreprocessor.py      |       10 |        2 |     80% |     10-11 |
| icenet\_mp/data\_processors/preprocessors/null.py               |        5 |        1 |     80% |         9 |
| icenet\_mp/data\_processors/sources/\_\_init\_\_.py             |       22 |        0 |    100% |           |
| icenet\_mp/data\_processors/sources/argo.py                     |       88 |       12 |     86% |60-61, 104-105, 136-141, 233-239 |
| icenet\_mp/data\_processors/sources/ftp.py                      |       40 |        3 |     92% |     66-68 |
| icenet\_mp/exceptions.py                                        |        3 |        0 |    100% |           |
| icenet\_mp/geotools/\_\_init\_\_.py                             |       10 |        0 |    100% |           |
| icenet\_mp/geotools/geographic\_field.py                        |       37 |       20 |     46% |15-16, 22, 27, 31, 40, 47, 60-71, 84 |
| icenet\_mp/geotools/geographic\_grid.py                         |       76 |       20 |     74% |31, 40, 46, 48, 54, 60, 65-66, 70-72, 81-83, 90, 94, 98, 102, 106, 110 |
| icenet\_mp/geotools/geographic\_metadata.py                     |       88 |       39 |     56% |14-15, 21, 26, 31, 36, 41, 45, 49-56, 60, 64, 68, 72, 76, 87-97, 101, 105, 109, 113, 117, 121, 128 |
| icenet\_mp/geotools/grid\_factory.py                            |       44 |       12 |     73% |17-18, 27-30, 35-38, 59-60 |
| icenet\_mp/geotools/reproject.py                                |       26 |        4 |     85% |36-37, 40-41 |
| icenet\_mp/losses/\_\_init\_\_.py                               |        4 |        0 |    100% |           |
| icenet\_mp/losses/rmse\_loss.py                                 |        9 |        1 |     89% |        15 |
| icenet\_mp/losses/weighted\_bce\_loss.py                        |       17 |       10 |     41% |26-33, 52-57 |
| icenet\_mp/losses/weighted\_l1\_loss.py                         |       17 |       10 |     41% |25-32, 51-56 |
| icenet\_mp/losses/weighted\_mse\_loss.py                        |       17 |       10 |     41% |26-33, 52-57 |
| icenet\_mp/metrics/\_\_init\_\_.py                              |        5 |        0 |    100% |           |
| icenet\_mp/metrics/daily\_metrics.py                            |       42 |        7 |     83% |30, 63-67, 81 |
| icenet\_mp/metrics/icenet\_accuracy.py                          |       24 |        3 |     88% |46, 51, 55 |
| icenet\_mp/metrics/sie\_error.py                                |       17 |       10 |     41% |27-31, 51-55, 59 |
| icenet\_mp/metrics/sie\_error\_abs.py                           |       26 |        2 |     92% |    70, 76 |
| icenet\_mp/model\_service.py                                    |      130 |       60 |     54% |34-37, 43-45, 59-60, 103-104, 119-121, 142-143, 157-158, 164-169, 183-257, 262-273, 281-294 |
| icenet\_mp/models/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| icenet\_mp/models/base\_model.py                                |       80 |        4 |     95% |94-95, 104, 108 |
| icenet\_mp/models/common/\_\_init\_\_.py                        |       13 |        0 |    100% |           |
| icenet\_mp/models/common/activations.py                         |        2 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_common.py                 |        8 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_downsample.py             |       10 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_upsample.py               |       13 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_norm\_act.py                     |        9 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_norm\_act\_upsample.py           |       10 |        0 |    100% |           |
| icenet\_mp/models/common/normalisations.py                      |       11 |        7 |     36% |     11-21 |
| icenet\_mp/models/common/normalised\_fold.py                    |       12 |        0 |    100% |           |
| icenet\_mp/models/common/patchembed.py                          |       13 |        0 |    100% |           |
| icenet\_mp/models/common/permute.py                             |        7 |        0 |    100% |           |
| icenet\_mp/models/common/resizing\_interpolation.py             |       13 |        0 |    100% |           |
| icenet\_mp/models/common/restrict\_range.py                     |       14 |        2 |     86% |     28-29 |
| icenet\_mp/models/common/shift.py                               |       14 |        0 |    100% |           |
| icenet\_mp/models/common/time\_embed.py                         |        9 |        4 |     56% | 26-30, 37 |
| icenet\_mp/models/common/transformerblock.py                    |       12 |        0 |    100% |           |
| icenet\_mp/models/common/weighted\_upsample.py                  |       15 |        0 |    100% |           |
| icenet\_mp/models/ddpm.py                                       |       96 |       79 |     18% |21-22, 38, 80-152, 155-156, 168-189, 205-245, 270-308, 334-358, 386-403 |
| icenet\_mp/models/decoders/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| icenet\_mp/models/decoders/base\_decoder.py                     |       14 |        2 |     86% |     40-41 |
| icenet\_mp/models/decoders/cnn\_decoder.py                      |       40 |        2 |     95% |     76-77 |
| icenet\_mp/models/decoders/naive\_linear\_decoder.py            |       19 |        0 |    100% |           |
| icenet\_mp/models/decoders/piecewise\_decoder.py                |       24 |        0 |    100% |           |
| icenet\_mp/models/diffusion/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| icenet\_mp/models/diffusion/gaussian\_diffusion.py              |       54 |       43 |     20% |39-72, 91-96, 115-135, 151-154, 175-180, 199-220 |
| icenet\_mp/models/diffusion/unet\_diffusion.py                  |       77 |       68 |     12% |56-178, 200-240, 256-271, 284-287 |
| icenet\_mp/models/encode\_process\_decode.py                    |       30 |        5 |     83% |43-45, 52-53 |
| icenet\_mp/models/encoders/\_\_init\_\_.py                      |        6 |        0 |    100% |           |
| icenet\_mp/models/encoders/base\_encoder.py                     |       24 |        2 |     92% |     59-60 |
| icenet\_mp/models/encoders/cnn\_encoder.py                      |       25 |        0 |    100% |           |
| icenet\_mp/models/encoders/naive\_linear\_encoder.py            |       15 |        0 |    100% |           |
| icenet\_mp/models/encoders/piecewise\_encoder.py                |       17 |        0 |    100% |           |
| icenet\_mp/models/encoders/reprojection\_encoder.py             |       33 |        0 |    100% |           |
| icenet\_mp/models/persistence.py                                |       16 |        0 |    100% |           |
| icenet\_mp/models/processors/\_\_init\_\_.py                    |        5 |        0 |    100% |           |
| icenet\_mp/models/processors/base\_processor.py                 |       18 |        0 |    100% |           |
| icenet\_mp/models/processors/null.py                            |       10 |        0 |    100% |           |
| icenet\_mp/models/processors/unet.py                            |       53 |        0 |    100% |           |
| icenet\_mp/models/processors/vit.py                             |       36 |        0 |    100% |           |
| icenet\_mp/types/\_\_init\_\_.py                                |        6 |        0 |    100% |           |
| icenet\_mp/types/complex\_datatypes.py                          |       41 |        3 |     93% | 63-64, 74 |
| icenet\_mp/types/enums.py                                       |       16 |        1 |     94% |        23 |
| icenet\_mp/types/protocols.py                                   |        4 |        0 |    100% |           |
| icenet\_mp/types/simple\_datatypes.py                           |       66 |        6 |     91% |   197-203 |
| icenet\_mp/types/typedefs.py                                    |       16 |        0 |    100% |           |
| icenet\_mp/utils.py                                             |       36 |       20 |     44% |12, 17-29, 34, 39-44, 57 |
| icenet\_mp/visualisations/\_\_init\_\_.py                       |        7 |        0 |    100% |           |
| icenet\_mp/visualisations/convert.py                            |       47 |        5 |     89% |82-84, 88-89 |
| icenet\_mp/visualisations/helpers.py                            |      152 |       30 |     80% |57, 59, 62-63, 79-80, 113, 133, 140, 144-146, 203-213, 317-318, 341-349, 372-378, 440, 476, 483, 495 |
| icenet\_mp/visualisations/land\_mask.py                         |       26 |        6 |     77% |     18-23 |
| icenet\_mp/visualisations/layout.py                             |      376 |       31 |     92% |217-218, 259-263, 303-304, 432, 655, 661, 675, 771-774, 776, 842-859, 875, 899, 914, 922-930, 1025, 1029, 1049 |
| icenet\_mp/visualisations/metadata.py                           |      192 |       24 |     88% |36, 40, 45, 59, 63, 68, 86-87, 119-125, 166, 170-174, 202-203, 251, 305, 326, 378, 380 |
| icenet\_mp/visualisations/plotter.py                            |       72 |       47 |     35% |39, 45-68, 74-99, 105-123, 133-155, 159-160 |
| icenet\_mp/visualisations/plotting\_core.py                     |      152 |       36 |     76% |59, 65-67, 87, 104, 127, 136-137, 149, 161, 169, 178, 279-285, 314-315, 334, 345-346, 381, 391-414, 450-454 |
| icenet\_mp/visualisations/plotting\_static.py                   |       65 |        5 |     92% |134-136, 237-238 |
| icenet\_mp/visualisations/plotting\_video.py                    |      117 |       17 |     85% |107-108, 110-113, 144-149, 186-188, 196-197, 340, 366-367, 445-449 |
| icenet\_mp/visualisations/range\_check.py                       |       77 |       16 |     79% |29, 33-36, 43-44, 53, 58-60, 98, 107, 144, 165, 171 |
| **TOTAL**                                                       | **4254** | **1143** | **73%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/alan-turing-institute/icenet-mp/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/icenet-mp/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/alan-turing-institute/icenet-mp/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/icenet-mp/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Falan-turing-institute%2Ficenet-mp%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/icenet-mp/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.