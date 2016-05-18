#!/bin/bash
for i in `seq 1 1 \`pdfinfo causal-bandits-nips.pdf|grep 'Pages'|cut -d: -f2|sed -e 's/ //g'\``
do
   echo Page $i;
   pdffonts -f $i -l $i causal-bandits-nips.pdf|grep 'Type 3';
done
