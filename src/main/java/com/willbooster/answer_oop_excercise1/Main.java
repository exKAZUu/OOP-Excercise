package com.willbooster.answer_oop_excercise1;

import com.willbooster.answer_oop_excercise1.ml.XorByNeuralNetwork;
import org.encog.Encog;
import org.encog.ml.data.basic.BasicMLDataSet;

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

        XorByNeuralNetwork xorByNeuralNetwork = new XorByNeuralNetwork();
        xorByNeuralNetwork.train(trainingSet);
        XorTester.test(trainingSet, xorByNeuralNetwork);

        Encog.getInstance().shutdown();
    }
}