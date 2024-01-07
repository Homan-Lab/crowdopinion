#!/bin/bash

declare dataset=MR
declare question_id=MR
declare date=Jun27

# ./run_language_experiments.sh "$dataset" "$question_id" "$date"
./bert_experiments.sh "$dataset" "$question_id" "$date"
