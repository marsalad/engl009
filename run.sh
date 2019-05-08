#!/bin/bash

if [ $# -lt 1 ]
	then 
		input=data 
	else
		input=$1
fi

if [ $# -lt 1 ]
	then 
		lab=labels.tsv 
	else
		lab=$2
fi

if [ $# -lt 3 ]
	then 
		output=analysis.tsv
	else
		output=$3
fi

python3 labels.py $input $lab
python3 analysis.py $lab $output