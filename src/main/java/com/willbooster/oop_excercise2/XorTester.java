package com.willbooster.oop_excercise2;

import com.willbooster.oop_excercise2.ml.XorByNeuralNetwork;
import com.willbooster.oop_excercise2.ml.XorPredictor;
import com.willbooster.oop_excercise2.ml.XorPredictorByRandom;
import org.encog.ml.data.MLDataSet;

public class XorTester {
    public static void test(MLDataSet trainingSet, XorByNeuralNetwork xorByNeuralNetwork) {
        var predictors = new XorPredictor[]{
                xorByNeuralNetwork,
                new XorPredictorByRandom(),
        };

        for (var predictor : predictors) {
            var allCount = 0;
            var correctCount = 0;
            for (var pair : trainingSet) {
                var output = predictor.predict(pair.getInput());
                if (output == pair.getIdeal().getData(0)) {
                    correctCount++;
                }
                allCount++;
            }

            System.out.println(predictor.getClass().getSimpleName() + ": " + correctCount + " / " + allCount);
        }
    }
}
