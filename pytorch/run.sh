#!/bin/sh

ARGS=""
if [[ "$1" == "--inference" ]]; then
    ARGS="$ARGS --inference"
    echo "### inference only"
    shift
fi

if [[ "$1" == "--single" ]]; then
    ARGS="$ARGS --single-batch-size"
    echo "### using single batch size"
    shift
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n"

python -u benchmark.py $ARGS

