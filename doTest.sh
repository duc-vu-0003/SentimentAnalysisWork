#!/bin/bash
clear
for ((a=0; a <= 9 ; a++))
do
   echo "Run test for fold$a."

   ./svm_light/svm_learn ./data/fold$a/train/feature_unigram ./result/fold$a/feature_unigram_model
   ./svm_light/svm_classify ./data/fold$a/test/feature_unigram ./result/fold$a/feature_unigram_model ./result/fold$a/predictions_unigram

  # ./svm_light/svm_learn ./data/fold$a/train/feature_bigram ./result/fold$a/feature_bigram_model
  # ./svm_light/svm_classify ./data/fold$a/test/feature_bigram ./result/fold$a/feature_bigram_model ./result/fold$a/predictions_bigram

  # ./svm_light/svm_learn ./data/fold$a/train/feature_top_unigram ./result/fold$a/feature_top_unigram_model
  # ./svm_light/svm_classify ./data/fold$a/test/feature_top_unigram ./result/fold$a/feature_top_unigram_model ./result/fold$a/predictions_top_unigram

   echo "--------------------------------"
   echo "--------------------------------"
done
