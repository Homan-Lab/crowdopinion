#!/bin/bash

declare dataset=MR
declare question_id=MR
declare date=Jan7

./bert_experiments.sh "$dataset" "$question_id" "$date"
