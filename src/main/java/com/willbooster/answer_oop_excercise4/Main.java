package com.willbooster.answer_oop_excercise4;

import com.willbooster.answer_oop_excercise4.ml.XorTrainerByNeuralNetwork;
import com.willbooster.answer_oop_excercise4.ml.XorTrainerBySVM;
import org.encog.Encog;
import org.encog.ml.data.basic.BasicMLDataSet;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Main {
    /**
     * XOR演算子を学習するための入力データ
     */
    public static double XOR_INPUT[][] = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
    };

    /**
     * 各入力データに対する教師データ（正解データ）
     */
    public static double XOR_IDEAL[][] = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
    };

    public static void main(final String args[]) {
        var trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        var trainers = new XorTrainer[]{
                new XorTrainerByNeuralNetwork(),
                new XorTrainerBySVM(),
        };

        var predictors = Arrays.stream(trainers).map(trainer -> trainer.train(trainingSet)).collect(Collectors.toList());
        XorTester.test(trainingSet, predictors);

        Encog.getInstance().shutdown();
    }
}