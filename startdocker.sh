#!/bin/bash

docker run -it -w /workdir -v $(pwd)/datasets:/workdir/datasets -v $(pwd)/src:/workdir/src tensorflow/tensorflow:1.2.0-py3 /bin/bash
