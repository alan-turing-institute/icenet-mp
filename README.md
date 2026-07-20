# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/icenet-mp/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                            |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| icenet\_mp/\_\_init\_\_.py                                      |        9 |        0 |    100% |           |
| icenet\_mp/callbacks/\_\_init\_\_.py                            |        6 |        0 |    100% |           |
| icenet\_mp/callbacks/activation\_saver.py                       |       92 |       70 |     24% |62-72, 82-115, 123-126, 134, 142, 145-161, 172-179, 191-217, 230-244, 252-257 |
| icenet\_mp/callbacks/ema\_weight\_averaging\_callback.py        |       21 |       12 |     43% |26-30, 36-37, 41-42, 48-54 |
| icenet\_mp/callbacks/metric\_summary\_callback.py               |       74 |       26 |     65% |28, 33, 55, 59-62, 70, 94-95, 102, 110-111, 115-118, 126-127, 133-138, 147-154, 160 |
| icenet\_mp/callbacks/plotting\_callback.py                      |      129 |       87 |     33% |81-84, 88-98, 104-119, 129-189, 204-222, 227-236, 249-271, 278-287 |
| icenet\_mp/callbacks/unconditional\_checkpoint.py               |       26 |       14 |     46% |18-20, 25, 30-31, 35-36, 40-52 |
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
| icenet\_mp/data\_loaders/common\_data\_module.py                |       99 |       33 |     67% |48-49, 92, 100-104, 109, 114, 119, 151, 160-162, 167, 175, 179-182, 188-204, 210-223, 229-245, 251-264 |
| icenet\_mp/data\_loaders/single\_dataset.py                     |      141 |        5 |     96% |179, 184, 208, 282-287 |
| icenet\_mp/data\_processors/\_\_init\_\_.py                     |        2 |        0 |    100% |           |
| icenet\_mp/data\_processors/data\_downloader.py                 |      186 |       57 |     69% |73-74, 111-151, 155-172, 201, 229-230, 240-241, 254, 272-275, 284-286, 295-296, 310-320 |
| icenet\_mp/data\_processors/data\_downloader\_factory.py        |       11 |        4 |     64% |     19-26 |
| icenet\_mp/data\_processors/filters/\_\_init\_\_.py             |       13 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/nan\_to\_num\_filter.py     |        9 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/reproject\_filter.py        |       31 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/set\_geography\_filter.py   |       19 |        0 |    100% |           |
| icenet\_mp/data\_processors/preprocessors/\_\_init\_\_.py       |        4 |        0 |    100% |           |
| icenet\_mp/data\_processors/preprocessors/icenet\_sic.py        |       64 |       46 |     28% |21-27, 31, 36-67, 75-132 |
| icenet\_mp/data\_processors/preprocessors/ipreprocessor.py      |       10 |        2 |     80% |     10-11 |
| icenet\_mp/data\_processors/preprocessors/null.py               |        5 |        1 |     80% |         9 |
| icenet\_mp/data\_processors/sources/\_\_init\_\_.py             |       22 |        0 |    100% |           |
| icenet\_mp/data\_processors/sources/argo.py                     |       89 |       12 |     87% |60-61, 105-106, 137-142, 234-240 |
| icenet\_mp/data\_processors/sources/ftp.py                      |       43 |        0 |    100% |           |
| icenet\_mp/data\_processors/sources/lazy\_argopy.py             |       13 |        2 |     85% |     29-30 |
| icenet\_mp/exceptions.py                                        |        3 |        0 |    100% |           |
| icenet\_mp/geotools/\_\_init\_\_.py                             |       10 |        0 |    100% |           |
| icenet\_mp/geotools/geographic\_field.py                        |       37 |       16 |     57% |22, 31, 40, 47, 60-71 |
| icenet\_mp/geotools/geographic\_grid.py                         |       76 |       19 |     75% |31, 40, 46, 48, 54, 60, 65-66, 70-72, 81-83, 90, 98, 102, 106, 110 |
| icenet\_mp/geotools/geographic\_metadata.py                     |       88 |       31 |     65% |26, 31, 36, 41, 45, 49-56, 60, 64, 68, 72, 76, 89, 93-97, 101, 105, 109, 113, 117, 121, 128 |
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
| icenet\_mp/model\_service.py                                    |      228 |      149 |     35% |36-39, 48-50, 64-65, 111-112, 127-129, 152-153, 167-168, 190-221, 225, 235-242, 247-252, 277-372, 377-388, 397-400, 420-449, 462-495, 501-563, 569-579, 590-621 |
| icenet\_mp/models/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| icenet\_mp/models/base\_model.py                                |       90 |        5 |     94% |99, 103, 150, 157-161 |
| icenet\_mp/models/common/\_\_init\_\_.py                        |       14 |        0 |    100% |           |
| icenet\_mp/models/common/activations.py                         |        2 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_common.py                 |        8 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_downsample.py             |       13 |        2 |     85% |     45-46 |
| icenet\_mp/models/common/conv\_block\_upsample.py               |       19 |        4 |     79% |55-56, 59-60 |
| icenet\_mp/models/common/conv\_norm\_act.py                     |        9 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_norm\_act\_upsample.py           |       10 |        0 |    100% |           |
| icenet\_mp/models/common/mask.py                                |       24 |        2 |     92% |     52-56 |
| icenet\_mp/models/common/normalisations.py                      |       11 |        7 |     36% |     11-21 |
| icenet\_mp/models/common/normalised\_fold.py                    |       19 |        0 |    100% |           |
| icenet\_mp/models/common/patchembed.py                          |       13 |        0 |    100% |           |
| icenet\_mp/models/common/permute.py                             |        7 |        0 |    100% |           |
| icenet\_mp/models/common/resizing\_interpolation.py             |       13 |        0 |    100% |           |
| icenet\_mp/models/common/restrict\_range.py                     |       14 |        0 |    100% |           |
| icenet\_mp/models/common/shift.py                               |       14 |        8 |     43% |10-14, 20-24 |
| icenet\_mp/models/common/time\_embed.py                         |        8 |        3 |     62% | 26-28, 35 |
| icenet\_mp/models/common/transformerblock.py                    |       12 |        0 |    100% |           |
| icenet\_mp/models/common/weighted\_upsample.py                  |       15 |        1 |     93% |        56 |
| icenet\_mp/models/ddpm.py                                       |      103 |       85 |     17% |22-23, 39, 90-184, 187-188, 200-222, 238-278, 303-341, 367-391, 419-436 |
| icenet\_mp/models/decoders/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| icenet\_mp/models/decoders/base\_decoder.py                     |       20 |        2 |     90% |     54-55 |
| icenet\_mp/models/decoders/cnn\_decoder.py                      |       35 |        2 |     94% |     73-74 |
| icenet\_mp/models/decoders/naive\_linear\_decoder.py            |       15 |        0 |    100% |           |
| icenet\_mp/models/decoders/piecewise\_decoder.py                |       28 |        2 |     93% |     77-82 |
| icenet\_mp/models/diffusion/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| icenet\_mp/models/diffusion/gaussian\_diffusion.py              |       54 |       43 |     20% |39-72, 91-96, 115-135, 151-154, 175-180, 199-220 |
| icenet\_mp/models/diffusion/unet\_diffusion.py                  |       78 |       68 |     13% |57-179, 201-241, 257-272, 285-288 |
| icenet\_mp/models/encode\_process\_decode.py                    |       35 |        8 |     77% |43-48, 64-70, 81-85 |
| icenet\_mp/models/encoders/\_\_init\_\_.py                      |        6 |        0 |    100% |           |
| icenet\_mp/models/encoders/base\_encoder.py                     |       24 |        2 |     92% |     57-58 |
| icenet\_mp/models/encoders/cnn\_encoder.py                      |       25 |        0 |    100% |           |
| icenet\_mp/models/encoders/naive\_linear\_encoder.py            |       15 |        0 |    100% |           |
| icenet\_mp/models/encoders/piecewise\_encoder.py                |       20 |        0 |    100% |           |
| icenet\_mp/models/encoders/reprojection\_encoder.py             |       33 |        0 |    100% |           |
| icenet\_mp/models/multistage/\_\_init\_\_.py                    |        4 |        0 |    100% |           |
| icenet\_mp/models/multistage/decoder\_stage.py                  |       44 |       25 |     43% |32-62, 80, 105-110, 121, 132-136 |
| icenet\_mp/models/multistage/encoder\_stage.py                  |       25 |        9 |     64% |29-48, 58, 72, 98-99, 107 |
| icenet\_mp/models/multistage/processor\_stage.py                |       65 |       44 |     32% |31-55, 72, 89-92, 102-103, 108-114, 139-175 |
| icenet\_mp/models/persistence.py                                |       17 |        0 |    100% |           |
| icenet\_mp/models/processors/\_\_init\_\_.py                    |        5 |        0 |    100% |           |
| icenet\_mp/models/processors/base\_processor.py                 |       25 |        2 |     92% |     34-38 |
| icenet\_mp/models/processors/null.py                            |       10 |        0 |    100% |           |
| icenet\_mp/models/processors/unet.py                            |       53 |        0 |    100% |           |
| icenet\_mp/models/processors/vit.py                             |       43 |        4 |     91% |41-42, 101-105 |
| icenet\_mp/types/\_\_init\_\_.py                                |        6 |        0 |    100% |           |
| icenet\_mp/types/complex\_datatypes.py                          |       75 |        9 |     88% |63-64, 74, 161-167 |
| icenet\_mp/types/enums.py                                       |       20 |        1 |     95% |        31 |
| icenet\_mp/types/protocols.py                                   |        4 |        0 |    100% |           |
| icenet\_mp/types/simple\_datatypes.py                           |       39 |        0 |    100% |           |
| icenet\_mp/types/typedefs.py                                    |       16 |        0 |    100% |           |
| icenet\_mp/utils.py                                             |       39 |       20 |     49% |13, 28-40, 45, 50-55, 68 |
| icenet\_mp/visualisations/\_\_init\_\_.py                       |        7 |        0 |    100% |           |
| icenet\_mp/visualisations/convert.py                            |       47 |        5 |     89% |82-84, 88-89 |
| icenet\_mp/visualisations/helpers.py                            |      151 |       23 |     85% |56, 58, 78-79, 111, 131, 138, 142-144, 200, 203-206, 315-316, 339-347, 438, 474, 481, 493 |
| icenet\_mp/visualisations/land\_mask.py                         |       23 |        4 |     83% |     15-18 |
| icenet\_mp/visualisations/layout.py                             |      376 |       31 |     92% |217-218, 259-263, 303-304, 432, 655, 661, 675, 771-774, 776, 842-859, 875, 899, 914, 922-930, 1025, 1029, 1049 |
| icenet\_mp/visualisations/metadata.py                           |      192 |       24 |     88% |36, 40, 45, 59, 63, 68, 86-87, 119-125, 166, 170-174, 202-203, 251, 305, 326, 378, 380 |
| icenet\_mp/visualisations/plotter.py                            |       78 |       54 |     31% |38, 48-72, 83-116, 126-145, 160-194, 201 |
| icenet\_mp/visualisations/plotting\_core.py                     |      148 |       36 |     76% |59, 65-67, 87, 104, 127, 136-137, 149, 161, 169, 178, 259-265, 294-295, 314, 325-326, 361, 371-394, 430-434 |
| icenet\_mp/visualisations/plotting\_static.py                   |       65 |        5 |     92% |132-134, 235-236 |
| icenet\_mp/visualisations/plotting\_video.py                    |      116 |       17 |     85% |106-107, 109-112, 142-147, 183-185, 193-194, 336, 362-363, 441-445 |
| icenet\_mp/visualisations/range\_check.py                       |       77 |       16 |     79% |29, 33-36, 43-44, 53, 58-60, 98, 107, 144, 165, 171 |
| **TOTAL**                                                       | **4712** | **1321** | **72%** |           |


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