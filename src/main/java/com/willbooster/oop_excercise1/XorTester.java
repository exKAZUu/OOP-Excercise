package com.willbooster.oop_excercise1;

import com.willbooster.oop_excercise1.ml.XorByNeuralNetwork;
import com.willbooster.oop_excercise1.ml.XorByRandom;
import com.willbooster.oop_excercise1.ml.XorOperator;
import org.encog.ml.data.MLDataSet;

public class XorTester {
    public static void test(MLDataSet trainingSet, XorByNeuralNetwork xorByNeuralNetwork) {
        var xors = new XorOperator[]{
                xorByNeuralNetwork,
                new XorByRandom(),
        };

        for (var xor : xors) {
            var allCount = 0;
            var correctCount = 0;
            for (var pair : trainingSet) {
                var output = xor.predict(pair.getInput());
                if (output == pair.getIdeal().getData(0)) {
                    correctCount++;
                }
                allCount++;
            }

            System.out.println(xor.getClass().getSimpleName() + ": " + correctCount + " / " + allCount);
        }
    }
}
