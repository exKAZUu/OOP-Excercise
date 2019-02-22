package com.willbooster.answer_oop_excercise4;

import com.willbooster.answer_oop_excercise4.ml.XorPredictorByRandom;
import org.encog.ml.data.MLDataSet;

import java.util.ArrayList;
import java.util.List;

public class XorTester {
    public static void test(MLDataSet trainingSet, List<XorPredictor> originalPredicors) {
        ArrayList<XorPredictor> predictors = new ArrayList<>(originalPredicors);
        predictors.add(new XorPredictorByRandom());

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