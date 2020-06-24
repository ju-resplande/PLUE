#!/bin/bash

FOLDER=$1

for FILE in ` find $FOLDER -name "*.tsv"`
do
    ftfy $FILE > $FILE.tmp
    mv $FILE.tmp $FILE
    echo $FILE
done
