#!/bin/bash

#Define input variables
pref="single"
data_type="trace"


#Define list
datapath="/mnlsc/data/MCBL4/dburrows/${pref}/"
array=($(find $datapath . -maxdepth 1 -name "*kmeans*$data_type*CCM*.h5*" ))

#Loop through and run kEDM


for i in "${array[@]}"
do
  echo "Running $i"
  filename=$i
  searchstring="run"
  rest=${filename#*$searchstring}
  savename="${filename:0:$(( ${#filename} - ${#rest} - ${#searchstring} + 6 ))}_${data_type}_CCMxmap.h5"

  edm-xmap -d, --dataset "data" --rho --rho-diff $filename $savename
  
  
done

echo "Finished!"

