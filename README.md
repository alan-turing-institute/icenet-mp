# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/icenet-mp/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                            |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| icenet\_mp/\_\_init\_\_.py                                      |        9 |        0 |    100% |           |
| icenet\_mp/callbacks/\_\_init\_\_.py                            |        6 |        0 |    100% |           |
| icenet\_mp/callbacks/activation\_saver.py                       |       92 |       70 |     24% |62-72, 82-115, 123-126, 134, 142, 145-161, 172-179, 191-217, 230-244, 252-257 |
| icenet\_mp/callbacks/ema\_weight\_averaging\_callback.py        |       13 |        8 |     38% |23-27, 33-39 |
| icenet\_mp/callbacks/metric\_summary\_callback.py               |       70 |       24 |     66% |28, 53, 57-60, 90-91, 98, 106-107, 111-114, 122-123, 129-134, 143-150, 156 |
| icenet\_mp/callbacks/plotting\_callback.py                      |      121 |       81 |     33% |78-81, 85-95, 101-116, 126-177, 192-210, 215-224, 237-259, 266-275 |
| icenet\_mp/callbacks/unconditional\_checkpoint.py               |       21 |       10 |     52% |17-19, 24, 29-30, 34-35, 39-40 |
| icenet\_mp/cli/\_\_init\_\_.py                                  |        2 |        0 |    100% |           |
| icenet\_mp/cli/datasets.py                                      |       38 |       20 |     47% |27-34, 47-53, 66-69, 73 |
| icenet\_mp/cli/evaluate.py                                      |       18 |        5 |     72% | 39-44, 48 |
| icenet\_mp/cli/hydra.py                                         |       29 |        0 |    100% |           |
| icenet\_mp/cli/main.py                                          |       23 |        8 |     65% | 39-51, 55 |
| icenet\_mp/cli/train.py                                         |       16 |        3 |     81% | 46-47, 54 |
| icenet\_mp/compatibility/\_\_init\_\_.py                        |       17 |        0 |    100% |           |
| icenet\_mp/compatibility/lightning/\_\_init\_\_.py              |        9 |        0 |    100% |           |
| icenet\_mp/compatibility/lightning/xpu\_accelerator.py          |       72 |       39 |     46% |35-42, 55, 60-62, 77-111, 125-127, 133, 144-147, 152 |
| icenet\_mp/compatibility/torch/\_\_init\_\_.py                  |        4 |        0 |    100% |           |
| icenet\_mp/compatibility/torch/patch\_interpolate\_antialias.py |       11 |        8 |     27% |     14-26 |
| icenet\_mp/compatibility/torch/patch\_open\_file\_limit.py      |       18 |        2 |     89% |     34-35 |
| icenet\_mp/compatibility/torch/patch\_parameter\_deepcopy.py    |       12 |        7 |     42% |     16-28 |
| icenet\_mp/config/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| icenet\_mp/data\_loaders/\_\_init\_\_.py                        |        4 |        0 |    100% |           |
| icenet\_mp/data\_loaders/combined\_dataset.py                   |       42 |        1 |     98% |        88 |
| icenet\_mp/data\_loaders/common\_data\_module.py                |       82 |       29 |     65% |47-48, 91, 99-103, 108, 113, 118, 123, 132, 136-139, 145-161, 167-180, 186-202, 208-221 |
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
| icenet\_mp/data\_processors/sources/argo.py                     |       87 |       12 |     86% |59-60, 103-104, 135-140, 232-238 |
| icenet\_mp/data\_processors/sources/ftp.py                      |       40 |        3 |     92% |     66-68 |
| icenet\_mp/data\_processors/sources/lazy\_argopy.py             |       13 |        2 |     85% |     29-30 |
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
| icenet\_mp/model\_service.py                                    |      211 |      131 |     38% |38-41, 47-49, 63-64, 108-109, 124-126, 148-149, 163-164, 186-204, 208, 218-224, 229-234, 259-346, 351-362, 371-374, 394-421, 434-468, 474-518, 524-534, 544-566 |
| icenet\_mp/models/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| icenet\_mp/models/base\_model.py                                |       90 |        5 |     94% |99, 103, 150, 157-161 |
| icenet\_mp/models/common/\_\_init\_\_.py                        |       13 |        0 |    100% |           |
| icenet\_mp/models/common/activations.py                         |        2 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_common.py                 |        8 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_downsample.py             |       13 |        2 |     85% |     42-43 |
| icenet\_mp/models/common/conv\_block\_upsample.py               |       19 |        4 |     79% |49-50, 53-54 |
| icenet\_mp/models/common/conv\_norm\_act.py                     |        9 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_norm\_act\_upsample.py           |       10 |        0 |    100% |           |
| icenet\_mp/models/common/normalisations.py                      |       11 |        7 |     36% |     11-21 |
| icenet\_mp/models/common/normalised\_fold.py                    |       19 |        1 |     95% |        41 |
| icenet\_mp/models/common/patchembed.py                          |       13 |        0 |    100% |           |
| icenet\_mp/models/common/permute.py                             |        7 |        0 |    100% |           |
| icenet\_mp/models/common/resizing\_interpolation.py             |       13 |        0 |    100% |           |
| icenet\_mp/models/common/restrict\_range.py                     |       14 |        2 |     86% |     28-29 |
| icenet\_mp/models/common/shift.py                               |       14 |        0 |    100% |           |
| icenet\_mp/models/common/time\_embed.py                         |        8 |        3 |     62% | 26-28, 35 |
| icenet\_mp/models/common/transformerblock.py                    |       12 |        0 |    100% |           |
| icenet\_mp/models/common/weighted\_upsample.py                  |       15 |        1 |     93% |        56 |
| icenet\_mp/models/ddpm.py                                       |       96 |       79 |     18% |21-22, 38, 80-152, 155-156, 168-189, 205-245, 270-308, 334-358, 386-403 |
| icenet\_mp/models/decoders/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| icenet\_mp/models/decoders/base\_decoder.py                     |       14 |        2 |     86% |     38-39 |
| icenet\_mp/models/decoders/cnn\_decoder.py                      |       39 |        2 |     95% |     77-78 |
| icenet\_mp/models/decoders/naive\_linear\_decoder.py            |       19 |        0 |    100% |           |
| icenet\_mp/models/decoders/piecewise\_decoder.py                |       24 |        0 |    100% |           |
| icenet\_mp/models/diffusion/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| icenet\_mp/models/diffusion/gaussian\_diffusion.py              |       54 |       43 |     20% |39-72, 91-96, 115-135, 151-154, 175-180, 199-220 |
| icenet\_mp/models/diffusion/unet\_diffusion.py                  |       77 |       68 |     12% |56-178, 200-240, 256-271, 284-287 |
| icenet\_mp/models/encode\_process\_decode.py                    |       30 |        5 |     83% |42-44, 55-59 |
| icenet\_mp/models/encoders/\_\_init\_\_.py                      |        6 |        0 |    100% |           |
| icenet\_mp/models/encoders/base\_encoder.py                     |       24 |        2 |     92% |     57-58 |
| icenet\_mp/models/encoders/cnn\_encoder.py                      |       25 |        0 |    100% |           |
| icenet\_mp/models/encoders/naive\_linear\_encoder.py            |       15 |        0 |    100% |           |
| icenet\_mp/models/encoders/piecewise\_encoder.py                |       17 |        0 |    100% |           |
| icenet\_mp/models/encoders/reprojection\_encoder.py             |       33 |        0 |    100% |           |
| icenet\_mp/models/multistage/\_\_init\_\_.py                    |        4 |        0 |    100% |           |
| icenet\_mp/models/multistage/decoder\_stage.py                  |       36 |       20 |     44% |30-60, 76, 101-106, 117 |
| icenet\_mp/models/multistage/encoder\_stage.py                  |       24 |        8 |     67% |29-44, 53, 66, 93, 101 |
| icenet\_mp/models/multistage/processor\_stage.py                |       57 |       40 |     30% |29-61, 77, 93-96, 106-107, 132-168 |
| icenet\_mp/models/persistence.py                                |       16 |        0 |    100% |           |
| icenet\_mp/models/processors/\_\_init\_\_.py                    |        5 |        0 |    100% |           |
| icenet\_mp/models/processors/base\_processor.py                 |       19 |        0 |    100% |           |
| icenet\_mp/models/processors/null.py                            |       10 |        0 |    100% |           |
| icenet\_mp/models/processors/unet.py                            |       53 |        0 |    100% |           |
| icenet\_mp/models/processors/vit.py                             |       40 |        2 |     95% |     92-96 |
| icenet\_mp/types/\_\_init\_\_.py                                |        6 |        0 |    100% |           |
| icenet\_mp/types/complex\_datatypes.py                          |       75 |        9 |     88% |63-64, 74, 161-167 |
| icenet\_mp/types/enums.py                                       |       16 |        1 |     94% |        23 |
| icenet\_mp/types/protocols.py                                   |        4 |        0 |    100% |           |
| icenet\_mp/types/simple\_datatypes.py                           |       35 |        0 |    100% |           |
| icenet\_mp/types/typedefs.py                                    |       16 |        0 |    100% |           |
| icenet\_mp/utils.py                                             |       36 |       20 |     44% |12, 17-29, 34, 39-44, 57 |
| icenet\_mp/visualisations/\_\_init\_\_.py                       |        7 |        0 |    100% |           |
| icenet\_mp/visualisations/convert.py                            |       47 |        5 |     89% |82-84, 88-89 |
| icenet\_mp/visualisations/helpers.py                            |      151 |       29 |     81% |56, 58, 61-62, 78-79, 111, 131, 138, 142-144, 200, 203-206, 315-316, 339-347, 370-376, 438, 474, 481, 493 |
| icenet\_mp/visualisations/land\_mask.py                         |       26 |        6 |     77% |     18-23 |
| icenet\_mp/visualisations/layout.py                             |      376 |       31 |     92% |217-218, 259-263, 303-304, 432, 655, 661, 675, 771-774, 776, 842-859, 875, 899, 914, 922-930, 1025, 1029, 1049 |
| icenet\_mp/visualisations/metadata.py                           |      192 |       24 |     88% |36, 40, 45, 59, 63, 68, 86-87, 119-125, 166, 170-174, 202-203, 251, 305, 326, 378, 380 |
| icenet\_mp/visualisations/plotter.py                            |       80 |       55 |     31% |39, 49-73, 84-117, 127-146, 161-195, 199-200 |
| icenet\_mp/visualisations/plotting\_core.py                     |      148 |       36 |     76% |59, 65-67, 87, 104, 127, 136-137, 149, 161, 169, 178, 259-265, 294-295, 314, 325-326, 361, 371-394, 430-434 |
| icenet\_mp/visualisations/plotting\_static.py                   |       65 |        5 |     92% |132-134, 235-236 |
| icenet\_mp/visualisations/plotting\_video.py                    |      116 |       17 |     85% |106-107, 109-112, 142-147, 183-185, 193-194, 336, 362-363, 441-445 |
| icenet\_mp/visualisations/range\_check.py                       |       77 |       16 |     79% |29, 33-36, 43-44, 53, 58-60, 98, 107, 144, 165, 171 |
| **TOTAL**                                                       | **4544** | **1317** | **71%** |           |


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