#!/bin/bash
clear
for ((a=0; a <= 9 ; a++))
do
   echo "Run test for fold$a."
   ./svm_light/svm_learn ./data/fold$a/train/feature_unigram ./result/fold$a/feature_unigram_model

   ./svm_light/svm_classify ./data/fold$a/test/feature_unigram ./result/fold$a/feature_unigram_model ./result/fold$a/predictions_unigram
done
