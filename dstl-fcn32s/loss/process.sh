#!/bin/bash
if [[ $# -eq 0 ]]; then
    cat ../train_dstl32.log | sed -n "/, loss/p" | awk -F" " '{if($12=="=") print $13; else if($13=="=") print $14; else if($14=="=") print $15; else if($15=="=") print $16; else if($16=="=") print $17; else if($17=="=") print $18; else if($18=="=") print $19; else if($19=="=") print $20; else if($20=="=") print $21; else if($21=="=") print $22}' | sed  '/^$/d' | sed -n '/^[[:digit:]]/p' > loss.txt
    cat ../train_dstl32.log | sed -n '/#0/p' | awk -F" " '{print $11}' | sed -n '/^[[:digit:]]/p' > loss_sem.txt
    cat ../train_dstl32.log | sed -n '/#1/p' | awk -F" " '{print $11}' | sed -n '/^[[:digit:]]/p' > loss_geo.txt
else
    cat $1 | sed -n "/, loss/p" | awk -F" " '{if($12=="=") print $13; else if($13=="=") print $14; else if($14=="=") print $15; else if($15=="=") print $16; else if($16=="=") print $17; else if($17=="=") print $18; else if($18=="=") print $19; else if($19=="=") print $20; else if($20=="=") print $21; else if($21=="=") print $22}' | sed  '/^$/d' | sed -n '/^[[:digit:]]/p' > loss.txt
    cat $1 | sed -n '/#0/p' | awk -F" " '{print $11}' | sed -n '/^[[:digit:]]/p' > loss_sem.txt
    cat $1 | sed -n '/#1/p' | awk -F" " '{print $11}' > loss_geo.txt
fi

#num_line=$(echo `cat loss.txt | wc -l`)
#interval=200

#python plot.py $num_line $interval
#rm loss.txt
#rm loss_sem.txt
#rm loss_geo.txt
