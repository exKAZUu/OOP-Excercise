package com.willbooster.oop_excercise2_answer;

import com.willbooster.oop_excercise2_answer.ml.XorPredictor;
import com.willbooster.oop_excercise2_answer.ml.XorPredictorByNeuralNetwork;
import com.willbooster.oop_excercise2_answer.ml.XorPredictorByReturningConstant;
import org.encog.ml.data.MLDataSet;

public class XorTester {
    public static void test(MLDataSet trainingSet, XorPredictorByNeuralNetwork xorPredictorByNeuralNetwork) {
        var predictors = new XorPredictor[]{
                xorPredictorByNeuralNetwork,
                new XorPredictorByReturningConstant(),
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
