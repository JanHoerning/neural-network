package ai;

import trainingData.DataManager;
import trainingData.TestData;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        int[] layers = {3, 8, 2};
        Network n = new Network(layers);

        //------test-data------
        double[][] data = {
                {1.0, 1.0, 0.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 1.0},
                {0.0, 0.0, 1.0}
        };

        double[][] answer = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };

        for (int i = 0; i < data.length; i++) {
            DataManager.addData(new TestData(data[i],answer[i]));
        }

        //------possibility-to-save-data-in-file------
        n.train(1000, 3, 0.3);

        //------data-saved-compleately-in-ram------
        //n.train(data,answer,1000,3,0.3);

        for (int i = 0; i < data.length; i++) {
            double[] prediction = n.predict(data[i]);
            System.out.println("Input: " + Arrays.toString(data[i]) + " Prediction: " + Arrays.toString(prediction) + " Expected: " + Arrays.toString(answer[i]));
        }
    }
}