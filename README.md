# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/icenet-mp/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                       |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------- | -------: | -------: | ------: | --------: |
| icenet\_mp/\_\_init\_\_.py                                 |        6 |        1 |     83% |         7 |
| icenet\_mp/callbacks/\_\_init\_\_.py                       |        6 |        0 |    100% |           |
| icenet\_mp/callbacks/ema\_weight\_averaging\_callback.py   |       13 |        8 |     38% |24-28, 34-40 |
| icenet\_mp/callbacks/metric\_summary\_callback.py          |       28 |       15 |     46% |25-27, 39-45, 54-61 |
| icenet\_mp/callbacks/plotting\_callback.py                 |       52 |       29 |     44% |    67-110 |
| icenet\_mp/callbacks/unconditional\_checkpoint.py          |       21 |       10 |     52% |17-19, 24, 29-30, 34-35, 39-40 |
| icenet\_mp/callbacks/wandb\_metric\_callback.py            |        8 |        3 |     62% |     11-13 |
| icenet\_mp/callbacks/weight\_averaging.py                  |       95 |       60 |     37% |96-112, 133, 149-161, 191-197, 212-217, 232-233, 248-249, 264-265, 277, 289, 310-326, 349-376, 387-398, 407-416 |
| icenet\_mp/cli/\_\_init\_\_.py                             |        2 |        0 |    100% |           |
| icenet\_mp/cli/datasets.py                                 |       55 |       28 |     49% |27-30, 37-40, 53-56, 66-69, 98-104, 118-121, 125 |
| icenet\_mp/cli/evaluate.py                                 |       16 |        3 |     81% | 27-28, 32 |
| icenet\_mp/cli/hydra.py                                    |       29 |        3 |     90% |     39-41 |
| icenet\_mp/cli/main.py                                     |       24 |        8 |     67% | 35-47, 51 |
| icenet\_mp/cli/train.py                                    |       14 |        3 |     79% | 20-21, 25 |
| icenet\_mp/config/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| icenet\_mp/data\_loaders/\_\_init\_\_.py                   |        4 |        0 |    100% |           |
| icenet\_mp/data\_loaders/combined\_dataset.py              |       50 |        9 |     82% |43-44, 93-99, 103, 115 |
| icenet\_mp/data\_loaders/common\_data\_module.py           |       60 |       38 |     37% |25-75, 88, 96, 104-106, 112-132, 138-158, 164-184, 190-210 |
| icenet\_mp/data\_loaders/single\_dataset.py                |       94 |        5 |     95% |   138-142 |
| icenet\_mp/data\_processors/\_\_init\_\_.py                |        2 |        0 |    100% |           |
| icenet\_mp/data\_processors/data\_downloader.py            |      200 |       74 |     63% |52-78, 82-84, 93-105, 109-139, 166-171, 184-185, 202-203, 209-215, 277-299, 303, 309-315, 328-339, 354-358, 370-371, 443 |
| icenet\_mp/data\_processors/data\_downloader\_factory.py   |       11 |        4 |     64% |     19-26 |
| icenet\_mp/data\_processors/filters/\_\_init\_\_.py        |       12 |        0 |    100% |           |
| icenet\_mp/data\_processors/filters/doubling\_filter.py    |       12 |        5 |     58% |16-17, 21-23 |
| icenet\_mp/data\_processors/filters/nan\_to\_num.py        |        9 |        2 |     78% |    13, 17 |
| icenet\_mp/data\_processors/preprocessors/\_\_init\_\_.py  |        4 |        0 |    100% |           |
| icenet\_mp/data\_processors/preprocessors/icenet\_sic.py   |       63 |       46 |     27% |20-26, 30, 35-66, 74-131 |
| icenet\_mp/data\_processors/preprocessors/ipreprocessor.py |        9 |        0 |    100% |           |
| icenet\_mp/data\_processors/preprocessors/null.py          |        5 |        1 |     80% |         9 |
| icenet\_mp/data\_processors/sources/\_\_init\_\_.py        |        9 |        0 |    100% |           |
| icenet\_mp/data\_processors/sources/ftp.py                 |       31 |        0 |    100% |           |
| icenet\_mp/exceptions.py                                   |        3 |        0 |    100% |           |
| icenet\_mp/model\_service.py                               |      123 |       52 |     58% |76-77, 92-94, 114-115, 129-130, 136-158, 163-182, 186, 193, 206-232, 237-257, 265-281 |
| icenet\_mp/models/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| icenet\_mp/models/base\_model.py                           |       54 |        0 |    100% |           |
| icenet\_mp/models/common/\_\_init\_\_.py                   |        9 |        0 |    100% |           |
| icenet\_mp/models/common/activations.py                    |        2 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_common.py            |        8 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_downsample.py        |       11 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_upsample.py          |       13 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_block\_upsample\_naive.py   |        8 |        0 |    100% |           |
| icenet\_mp/models/common/conv\_norm\_act.py                |       20 |        4 |     80% | 41-46, 73 |
| icenet\_mp/models/common/patchembed.py                     |       13 |        8 |     38% |19-24, 38-40 |
| icenet\_mp/models/common/resizing\_interpolation.py        |       13 |        0 |    100% |           |
| icenet\_mp/models/common/time\_embed.py                    |        9 |        4 |     56% | 26-30, 37 |
| icenet\_mp/models/common/transformerblock.py               |       12 |        7 |     42% |16-22, 40-42 |
| icenet\_mp/models/ddpm.py                                  |      119 |       94 |     21% |17, 35-36, 52, 94-194, 197-198, 215-236, 244-246, 262-302, 325-356, 381-405, 434-457 |
| icenet\_mp/models/decoders/\_\_init\_\_.py                 |        4 |        0 |    100% |           |
| icenet\_mp/models/decoders/base\_decoder.py                |       14 |        2 |     86% |     40-41 |
| icenet\_mp/models/decoders/cnn\_decoder.py                 |       40 |        0 |    100% |           |
| icenet\_mp/models/decoders/naive\_linear\_decoder.py       |       19 |        0 |    100% |           |
| icenet\_mp/models/diffusion/\_\_init\_\_.py                |        3 |        0 |    100% |           |
| icenet\_mp/models/diffusion/gaussian\_diffusion.py         |       54 |       43 |     20% |39-72, 91-96, 115-135, 151-154, 175-180, 199-220 |
| icenet\_mp/models/diffusion/unet\_diffusion.py             |       77 |       68 |     12% |56-177, 199-239, 255-270, 283-286 |
| icenet\_mp/models/encode\_process\_decode.py               |       21 |        0 |    100% |           |
| icenet\_mp/models/encoders/\_\_init\_\_.py                 |        4 |        0 |    100% |           |
| icenet\_mp/models/encoders/base\_encoder.py                |       14 |        2 |     86% |     44-45 |
| icenet\_mp/models/encoders/cnn\_encoder.py                 |       25 |        0 |    100% |           |
| icenet\_mp/models/encoders/naive\_linear\_encoder.py       |       15 |        0 |    100% |           |
| icenet\_mp/models/losses/\_\_init\_\_.py                   |        4 |        0 |    100% |           |
| icenet\_mp/models/losses/weighted\_bce\_loss.py            |        8 |        3 |     62% | 23, 37-42 |
| icenet\_mp/models/losses/weighted\_l1\_loss.py             |       10 |        4 |     60% | 23, 36-38 |
| icenet\_mp/models/losses/weighted\_mse\_loss.py            |       11 |        6 |     45% | 23, 42-47 |
| icenet\_mp/models/metrics/\_\_init\_\_.py                  |        3 |        0 |    100% |           |
| icenet\_mp/models/metrics/icenet\_accuracy.py              |       18 |       11 |     39% |18-23, 31-41, 47 |
| icenet\_mp/models/metrics/sie\_error.py                    |       17 |       10 |     41% |27-31, 51-55, 59 |
| icenet\_mp/models/persistence.py                           |       16 |        0 |    100% |           |
| icenet\_mp/models/processors/\_\_init\_\_.py               |        5 |        0 |    100% |           |
| icenet\_mp/models/processors/base\_processor.py            |       18 |        0 |    100% |           |
| icenet\_mp/models/processors/null.py                       |       10 |        0 |    100% |           |
| icenet\_mp/models/processors/unet.py                       |       53 |        0 |    100% |           |
| icenet\_mp/models/processors/vit.py                        |       36 |       27 |     25% |34-67, 81-106 |
| icenet\_mp/plugins.py                                      |       10 |        0 |    100% |           |
| icenet\_mp/types/\_\_init\_\_.py                           |        6 |        0 |    100% |           |
| icenet\_mp/types/complex\_datatypes.py                     |       36 |       12 |     67% |52-59, 63-65, 69 |
| icenet\_mp/types/enums.py                                  |        7 |        0 |    100% |           |
| icenet\_mp/types/protocols.py                              |        4 |        0 |    100% |           |
| icenet\_mp/types/simple\_datatypes.py                      |       59 |        0 |    100% |           |
| icenet\_mp/types/typedefs.py                               |       13 |        0 |    100% |           |
| icenet\_mp/utils.py                                        |       28 |       16 |     43% |9, 14-26, 31, 44 |
| icenet\_mp/visualisations/\_\_init\_\_.py                  |        3 |        0 |    100% |           |
| icenet\_mp/visualisations/convert.py                       |       47 |        5 |     89% |82-84, 88-89 |
| icenet\_mp/visualisations/helpers.py                       |      151 |       27 |     82% |57, 59, 62-63, 79-80, 113, 133, 140, 145-146, 206-221, 332-333, 356-364, 387-393, 455, 488 |
| icenet\_mp/visualisations/land\_mask.py                    |       26 |        6 |     77% |     18-23 |
| icenet\_mp/visualisations/layout.py                        |      372 |       31 |     92% |217-218, 259-263, 303-304, 432, 655, 661, 675, 771-774, 776, 834-851, 863, 887, 902, 910-918, 1013, 1017, 1037 |
| icenet\_mp/visualisations/metadata.py                      |      192 |       24 |     88% |36, 40, 45, 59, 63, 68, 86-87, 119-125, 166, 170-174, 202-203, 251, 304, 325, 377, 379 |
| icenet\_mp/visualisations/plotter.py                       |       72 |       46 |     36% |29-30, 41-64, 70-95, 101-119, 129-151 |
| icenet\_mp/visualisations/plotting\_core.py                |      165 |       48 |     71% |59, 65-67, 87, 104, 127, 136-137, 147-149, 161, 169-180, 279-285, 314-315, 334, 345-346, 381, 391-414, 451-454, 480-484 |
| icenet\_mp/visualisations/plotting\_static.py              |       64 |        5 |     92% |129-131, 230-231 |
| icenet\_mp/visualisations/plotting\_video.py               |      117 |       17 |     85% |104-105, 107-110, 141-146, 180-182, 190-191, 334, 360-361, 440-444 |
| icenet\_mp/visualisations/range\_check.py                  |       77 |       16 |     79% |29, 33-36, 43-44, 53, 58-60, 98, 107, 144, 165, 171 |
| icenet\_mp/xpu/\_\_init\_\_.py                             |        9 |        0 |    100% |           |
| icenet\_mp/xpu/accelerator.py                              |       72 |       39 |     46% |35-42, 55, 60-62, 77-111, 125-127, 133, 144-147, 152 |
| **TOTAL**                                                  | **3395** |  **992** | **71%** |           |


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