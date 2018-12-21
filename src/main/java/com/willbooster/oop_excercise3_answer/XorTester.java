package com.willbooster.oop_excercise3_answer;

import com.willbooster.oop_excercise3_answer.ml.XorPredictor;
import com.willbooster.oop_excercise3_answer.ml.XorPredictorByReturningConstant;
import org.encog.ml.data.MLDataSet;

import java.util.ArrayList;
import java.util.List;

public class XorTester {
    public static void test(MLDataSet trainingSet, List<XorPredictor> originalPredictors) {
        ArrayList<XorPredictor> predictors = new ArrayList<>(originalPredictors);
        predictors.add(new XorPredictorByReturningConstant());

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
