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
| icenet\_mp/cli/evaluate.py                                      |       18 |        5 |     72% | 37-42, 46 |
| icenet\_mp/cli/hydra.py                                         |       29 |        0 |    100% |           |
| icenet\_mp/cli/main.py                                          |       23 |        8 |     65% | 40-52, 56 |
| icenet\_mp/cli/train.py                                         |       16 |        3 |     81% | 45-46, 53 |
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
| icenet\_mp/data\_loaders/single\_dataset.py                     |      141 |        5 |     96% |185, 190, 214, 288-293 |
| icenet\_mp/data\_processors/\_\_init\_\_.py                     |        2 |        0 |    100% |           |
| icenet\_mp/data\_processors/data\_downloader.py                 |      199 |       59 |     70% |73-74, 111-151, 155-172, 201, 210-211, 244-245, 255-256, 269, 287-290, 299-301, 310-311, 325-335 |
| icenet\_mp/data\_processors/data\_downloader\_factory.py        |       11 |        4 |     64% |     19-26 |
| icenet\_mp/data\_processors/filters/\_\_init\_\_.py             |       13 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/nan\_to\_num\_filter.py     |        9 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/reproject\_filter.py        |       31 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/set\_geography\_filter.py   |       19 |        0 |    100% |           |
| icenet\_mp/data\_processors/preprocessors/\_\_init\_\_.py       |        4 |        0 |    100% |           |
| icenet\_mp/data\_processors/preprocessors/icenet\_sic.py        |       64 |       46 |     28% |21-27, 31, 36-67, 75-132 |
| icenet\_mp/data\_processors/preprocessors/ipreprocessor.py      |       10 |        2 |     80% |     10-11 |
| icenet\_mp/data\_processors/preprocessors/null.py               |        5 |        1 |     80% |         9 |
| icenet\_mp/data\_processors/sources/\_\_init\_\_.py             |       24 |        0 |    100% |           |
| icenet\_mp/data\_processors/sources/argo.py                     |       89 |       12 |     87% |60-61, 105-106, 137-142, 234-240 |
| icenet\_mp/data\_processors/sources/ftp.py                      |       43 |        0 |    100% |           |
| icenet\_mp/data\_processors/sources/lazy\_argopy.py             |       13 |        2 |     85% |     29-30 |
| icenet\_mp/data\_processors/sources/synthetic.py                |       27 |       10 |     63% |33-42, 50-88 |
| icenet\_mp/exceptions.py                                        |        3 |        0 |    100% |           |
| icenet\_mp/geotools/\_\_init\_\_.py                             |       10 |        0 |    100% |           |
| icenet\_mp/geotools/geographic\_field.py                        |       37 |       16 |     57% |22, 31, 40, 47, 60-71 |
| icenet\_mp/geotools/geographic\_grid.py                         |       76 |       19 |     75% |31, 40, 46, 48, 54, 60, 65-66, 70-72, 81-83, 90, 98, 102, 106, 110 |
| icenet\_mp/geotools/geographic\_metadata.py                     |       88 |       31 |     65% |26, 31, 36, 41, 45, 49-56, 60, 64, 68, 72, 76, 89, 93-97, 101, 105, 109, 113, 117, 121, 128 |
| icenet\_mp/geotools/grid\_factory.py                            |       44 |       12 |     73% |17-18, 27-30, 35-38, 59-60 |
| icenet\_mp/geotools/reproject.py                                |       26 |        4 |     85% |36-37, 40-41 |
| icenet\_mp/loggers/\_\_init\_\_.py                              |        2 |        2 |      0% |       3-5 |
| icenet\_mp/loggers/local\_file\_logger.py                       |       53 |       53 |      0% |    10-118 |
| icenet\_mp/losses/\_\_init\_\_.py                               |        4 |        0 |    100% |           |
| icenet\_mp/losses/rmse\_loss.py                                 |        9 |        1 |     89% |        15 |
| icenet\_mp/losses/weighted\_bce\_loss.py                        |       17 |       10 |     41% |26-33, 52-57 |
| icenet\_mp/losses/weighted\_l1\_loss.py                         |       17 |       10 |     41% |25-32, 51-56 |
| icenet\_mp/losses/weighted\_mse\_loss.py                        |       17 |       10 |     41% |26-33, 52-57 |
| icenet\_mp/metrics/\_\_init\_\_.py                              |        6 |        0 |    100% |           |
| icenet\_mp/metrics/centroid\_error.py                           |       23 |       16 |     30% |34-42, 47-55 |
| icenet\_mp/metrics/daily\_metrics.py                            |       46 |        5 |     89% |47-51, 59, 68, 82 |
| icenet\_mp/metrics/icenet\_accuracy.py                          |       24 |        3 |     88% |46, 51, 55 |
| icenet\_mp/metrics/sie\_error.py                                |       17 |       10 |     41% |27-31, 51-55, 59 |
| icenet\_mp/metrics/sie\_error\_abs.py                           |       26 |        2 |     92% |    70, 76 |
| icenet\_mp/model\_service.py                                    |      249 |      170 |     32% |37-40, 54-55, 102-103, 118-120, 143-144, 158-159, 181-212, 216, 237-265, 270-275, 300-407, 412-423, 432-435, 455-484, 497-534, 540-606, 612-622, 633-668 |
| icenet\_mp/models/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| icenet\_mp/models/base\_model.py                                |       93 |        5 |     95% |124, 128, 175, 182-186 |
| icenet\_mp/models/common/\_\_init\_\_.py                        |       21 |        0 |    100% |           |
| icenet\_mp/models/common/activations.py                         |        2 |        0 |    100% |           |
| icenet\_mp/models/common/channel\_adaptor.py                    |       18 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_common.py                 |        8 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_downsample.py             |       13 |        2 |     85% |     45-46 |
| icenet\_mp/models/common/conv\_block\_upsample.py               |       19 |        4 |     79% |55-56, 59-60 |
| icenet\_mp/models/common/conv\_norm\_act.py                     |        9 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_norm\_act\_upsample.py           |       10 |        0 |    100% |           |
| icenet\_mp/models/common/gated\_attention.py                    |       52 |       27 |     48% |23-46, 49-50, 92-116, 124-130, 135-136, 139-141 |
| icenet\_mp/models/common/glumb\_conv.py                         |       23 |        1 |     96% |        71 |
| icenet\_mp/models/common/lite\_mla.py                           |       30 |        2 |     93% |     44-45 |
| icenet\_mp/models/common/mask.py                                |       24 |        2 |     92% |     52-56 |
| icenet\_mp/models/common/normalisations.py                      |       20 |        9 |     55% |9-10, 13-14, 25, 34-38 |
| icenet\_mp/models/common/normalised\_fold.py                    |       19 |        0 |    100% |           |
| icenet\_mp/models/common/patchembed.py                          |       13 |        0 |    100% |           |
| icenet\_mp/models/common/permute.py                             |        7 |        0 |    100% |           |
| icenet\_mp/models/common/res\_block.py                          |       16 |        0 |    100% |           |
| icenet\_mp/models/common/residual\_downsample.py                |       20 |        5 |     75% | 54-59, 76 |
| icenet\_mp/models/common/residual\_upsample.py                  |       15 |        1 |     93% |        67 |
| icenet\_mp/models/common/resizing\_interpolation.py             |       13 |        0 |    100% |           |
| icenet\_mp/models/common/restrict\_range.py                     |       14 |        0 |    100% |           |
| icenet\_mp/models/common/shift.py                               |       14 |        8 |     43% |10-14, 20-24 |
| icenet\_mp/models/common/time\_embed.py                         |        8 |        3 |     62% | 26-28, 35 |
| icenet\_mp/models/common/transformerblock.py                    |       12 |        0 |    100% |           |
| icenet\_mp/models/common/weighted\_upsample.py                  |       16 |        0 |    100% |           |
| icenet\_mp/models/ddpm.py                                       |      147 |      124 |     16% |25-26, 42, 99-198, 201-202, 244-247, 280-297, 335-384, 400-440, 465-516, 541-564, 591-611 |
| icenet\_mp/models/decoders/\_\_init\_\_.py                      |        6 |        0 |    100% |           |
| icenet\_mp/models/decoders/base\_decoder.py                     |       20 |        2 |     90% |     56-57 |
| icenet\_mp/models/decoders/cnn\_decoder.py                      |       35 |        2 |     94% |     74-75 |
| icenet\_mp/models/decoders/deep\_compression\_decoder.py        |       41 |        6 |     85% |58-59, 61-62, 64-65 |
| icenet\_mp/models/decoders/naive\_linear\_decoder.py            |       15 |        0 |    100% |           |
| icenet\_mp/models/decoders/piecewise\_decoder.py                |       28 |        2 |     93% |     77-82 |
| icenet\_mp/models/diffusion/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| icenet\_mp/models/diffusion/gaussian\_diffusion.py              |       54 |       43 |     20% |39-72, 91-96, 115-135, 151-154, 175-180, 199-220 |
| icenet\_mp/models/diffusion/unet\_diffusion.py                  |       78 |       68 |     13% |58-180, 202-242, 258-273, 286-289 |
| icenet\_mp/models/encode\_process\_decode.py                    |       36 |        8 |     78% |48-53, 69-75, 86-90 |
| icenet\_mp/models/encoders/\_\_init\_\_.py                      |        7 |        0 |    100% |           |
| icenet\_mp/models/encoders/base\_encoder.py                     |       24 |        2 |     92% |     57-58 |
| icenet\_mp/models/encoders/cnn\_encoder.py                      |       25 |        0 |    100% |           |
| icenet\_mp/models/encoders/deep\_compression\_encoder.py        |       42 |        6 |     86% |59-60, 62-63, 65-66 |
| icenet\_mp/models/encoders/naive\_linear\_encoder.py            |       15 |        0 |    100% |           |
| icenet\_mp/models/encoders/piecewise\_encoder.py                |       20 |        0 |    100% |           |
| icenet\_mp/models/encoders/reprojection\_encoder.py             |       33 |        0 |    100% |           |
| icenet\_mp/models/multistage/\_\_init\_\_.py                    |        4 |        0 |    100% |           |
| icenet\_mp/models/multistage/decoder\_stage.py                  |       45 |       25 |     44% |35-65, 83, 108-113, 124, 135-139 |
| icenet\_mp/models/multistage/encoder\_stage.py                  |       25 |        9 |     64% |29-48, 59, 73, 99-100, 108 |
| icenet\_mp/models/multistage/processor\_stage.py                |       66 |       44 |     33% |37-61, 78, 95-98, 108-109, 114-120, 145-181 |
| icenet\_mp/models/persistence.py                                |       17 |        0 |    100% |           |
| icenet\_mp/models/processors/\_\_init\_\_.py                    |        6 |        0 |    100% |           |
| icenet\_mp/models/processors/base\_processor.py                 |       25 |        2 |     92% |     34-38 |
| icenet\_mp/models/processors/gsta.py                            |       22 |       12 |     45% |65-73, 99-110 |
| icenet\_mp/models/processors/null.py                            |       10 |        0 |    100% |           |
| icenet\_mp/models/processors/unet.py                            |       53 |        0 |    100% |           |
| icenet\_mp/models/processors/vit.py                             |       43 |        4 |     91% |41-42, 101-105 |
| icenet\_mp/synthetic/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| icenet\_mp/synthetic/debug\_video.py                            |       47 |       47 |      0% |     9-122 |
| icenet\_mp/synthetic/shapes.py                                  |       81 |        0 |    100% |           |
| icenet\_mp/synthetic/trajectories.py                            |       77 |        9 |     88% |180-191, 207-211 |
| icenet\_mp/types/\_\_init\_\_.py                                |        6 |        0 |    100% |           |
| icenet\_mp/types/complex\_datatypes.py                          |       76 |        9 |     88% |63-64, 74, 163-169 |
| icenet\_mp/types/enums.py                                       |       20 |        1 |     95% |        31 |
| icenet\_mp/types/protocols.py                                   |        4 |        0 |    100% |           |
| icenet\_mp/types/simple\_datatypes.py                           |       39 |        0 |    100% |           |
| icenet\_mp/types/typedefs.py                                    |       16 |        0 |    100% |           |
| icenet\_mp/utils.py                                             |       39 |       20 |     49% |13, 28-40, 45, 50-55, 68 |
| icenet\_mp/visualisations/\_\_init\_\_.py                       |        7 |        0 |    100% |           |
| icenet\_mp/visualisations/convert.py                            |       46 |        5 |     89% |76-78, 82-83 |
| icenet\_mp/visualisations/helpers.py                            |      151 |       23 |     85% |56, 58, 78-79, 111, 131, 138, 142-144, 200, 203-206, 315-316, 339-347, 438, 474, 481, 493 |
| icenet\_mp/visualisations/land\_mask.py                         |       23 |        4 |     83% |     15-18 |
| icenet\_mp/visualisations/layout.py                             |      376 |       31 |     92% |217-218, 259-263, 303-304, 432, 655, 661, 675, 771-774, 776, 842-859, 875, 899, 914, 922-930, 1025, 1029, 1049 |
| icenet\_mp/visualisations/metadata.py                           |      192 |       24 |     88% |36, 40, 45, 59, 63, 68, 86-87, 119-125, 166, 170-174, 202-203, 251, 305, 326, 378, 380 |
| icenet\_mp/visualisations/plotter.py                            |       78 |       54 |     31% |38, 48-72, 83-116, 126-145, 160-194, 201 |
| icenet\_mp/visualisations/plotting\_core.py                     |      148 |       36 |     76% |59, 65-67, 87, 104, 127, 136-137, 149, 161, 169, 178, 259-265, 294-295, 314, 325-326, 361, 371-394, 430-434 |
| icenet\_mp/visualisations/plotting\_static.py                   |       65 |        5 |     92% |132-134, 237-238 |
| icenet\_mp/visualisations/plotting\_video.py                    |      116 |       17 |     85% |106-107, 109-112, 142-147, 183-185, 193-194, 337, 363-364, 445-449 |
| icenet\_mp/visualisations/range\_check.py                       |       77 |       16 |     79% |29, 33-36, 43-44, 53, 58-60, 98, 107, 144, 165, 171 |
| **TOTAL**                                                       | **5414** | **1579** | **71%** |           |


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