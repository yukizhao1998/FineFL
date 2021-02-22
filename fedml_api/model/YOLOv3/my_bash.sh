# Set Up Image Lists
cd data
cd coco

for ((i=0; i<2; i++))do
paste <(awk "{print \"$PWD\"}" <'data_split/'$i'_'val.part) 'data_split/'$i'_'val.part | tr -d '\t' > ./data_split/$i'_'val.txt
paste <(awk "{print \"$PWD\"}" <'data_split/'$i'_'train.part) 'data_split/'$i'_'train.part | tr -d '\t' > ./data_split/$i'_'train.txt
done
#paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
#paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt