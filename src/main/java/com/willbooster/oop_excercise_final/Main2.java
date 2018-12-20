package com.willbooster.oop_excercise_final;

import com.willbooster.oop_excercise_final.ml.NeuralNetworkXorTrainer;
import com.willbooster.oop_excercise_final.ml.SvmXorTrainer;
import org.encog.Encog;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Main2 {

    /**
     * The input necessary for XOR.
     */
    public static double XOR_INPUT[][] = {{0.0, 0.0}, {1.0, 0.0},
            {0.0, 1.0}, {1.0, 1.0}};

    /**
     * The ideal data necessary for XOR.
     */
    public static double XOR_IDEAL[][] = {{0.0}, {1.0}, {1.0}, {0.0}};

    /**
     * The main method.
     *
     * @param args No arguments are used.
     */
    public static void main(final String args[]) {
        var trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        var trainers = new XorTrainer[]{
                new NeuralNetworkXorTrainer(),
                new SvmXorTrainer(),
        };

        var predictors = Arrays.stream(trainers).map(trainer -> trainer.train(trainingSet)).collect(Collectors.toList());

        predictors.forEach(predictor -> {
            trainingSet.forEach(pair -> {
                var output = predictor.predict(pair.getInput());
                System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                        + ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
            });
        });

        System.out.println(predictors.get(1).predict(new BasicMLData(new double[]{-1.0, 0.0})).getData(0));

        Encog.getInstance().shutdown();
    }
}