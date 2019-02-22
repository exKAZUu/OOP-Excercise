package com.willbooster.answer_oop_excercise1;

import com.willbooster.answer_oop_excercise1.ml.XorByNeuralNetwork;
import com.willbooster.answer_oop_excercise1.ml.XorPredictor;
import com.willbooster.answer_oop_excercise1.ml.XorPredictorByRandom;
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
