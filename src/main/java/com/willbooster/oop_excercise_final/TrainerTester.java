package com.willbooster.oop_excercise_final;

import org.encog.ml.data.MLDataSet;

public class TrainerTester {
    public static void test(MLDataSet trainingSet, XorPredictor[] predictors) {
        for (var predictor : predictors) {
            var allCount = 0;
            var correctCount = 0;
            for (var pair : trainingSet) {
                var output = predictor.predict(pair.getInput());
                if (output.getData(0) == pair.getIdeal().getData(0)) {
                    correctCount++;
                }
                allCount++;
            }

            System.out.println(predictor.getClass().getSimpleName() + ": " + correctCount + " / " + allCount);
        }
    }
}