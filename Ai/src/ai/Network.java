package ai;

import trainingData.Data;
import trainingData.DataManager;

import java.util.ArrayList;
import java.util.Collections;

public class Network{

    private Neuron[][] layers;
    private double[][] values;

    public Network(int[] layers) {
        this.layers = new Neuron[layers.length - 1][];
        for (int i = 0; i < this.layers.length; i++) {
            this.layers[i] = new Neuron[layers[i + 1]];
            for (int j = 0; j < layers[i + 1]; j++) {
                this.layers[i][j] = new Neuron(layers[i]);
            }
        }
    }

    public Network(double[][][] layers) {
        this.layers = new Neuron[layers.length][];
        for (int i = 0; i < this.layers.length; i++) {
            this.layers[i] = new Neuron[layers[i].length];
            for (int j = 0; j < layers[i].length; j++) {
                this.layers[i][j] = new Neuron(layers[i][j]);
            }
        }
    }

    public double[] predict(double[] in) {
        this.values = new double[layers.length][];
        for(int i = 0; i < layers.length; i++) {
            this.values[i] = new double[layers[i].length];
        }

        for(int i = 0; i < layers.length; i++) {
            for(int j = 0; j < layers[i].length; j++) {
                if(i == 0) {
                    this.values[i][j] = layers[i][j].compute(in);
                } else {
                    this.values[i][j] = layers[i][j].compute(values[i-1]);
                }
            }
        }

        return this.values[layers.length-1];
    }

    private ArrayList<BatchData> createMiniBatches(double[][] data, double[][] answers, int batchSize) {
        ArrayList<BatchData> batches = new ArrayList<>();

        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);

        for (int i = 0; i < data.length; i += batchSize) {
            int actualBatchSize = Math.min(batchSize, data.length - i);
            double[][] batchData = new double[actualBatchSize][];
            double[][] batchAnswers = new double[actualBatchSize][];

            for (int j = 0; j < actualBatchSize; j++) {
                int pos = indices.get(i + j);
                batchData[j] = data[pos];
                batchAnswers[j] = answers[pos];
            }

            batches.add(new BatchData(batchData, batchAnswers));
        }

        return batches;
    }

    private ArrayList<Data> loadNextBatche(int bSize, int temp) {
        ArrayList<Data> batch = new ArrayList<>();

        for (int i = 0; i < bSize; i ++) {
            int pos = (temp + i) % DataManager.getSize();
            if((temp + i) % DataManager.getSize() == 0) {
                DataManager.shuffle();
            }

            batch.add(DataManager.getData(pos));
        }

        return batch;
    }

    public void train(double[][] data, double[][] answers, int epochs, int batchSize, double learningRate) {
        for(int epoch = 0; epoch < epochs; epoch++) {
            ArrayList<BatchData> batches = createMiniBatches(data, answers, batchSize);
            double epochLoss = 0;

            for(BatchData batch : batches) {
                double[][][] batchGradients = new double[layers.length][][];
                double[][] batchBiasGradients = new double[layers.length][];

                for (int i = 0; i < layers.length; i++) {
                    batchGradients[i] = new double[layers[i].length][];
                    batchBiasGradients[i] = new double[layers[i].length];
                    for (int j = 0; j < layers[i].length; j++) {
                        batchGradients[i][j] = new double[layers[i][j].getWeigth().length];
                    }
                }

                double batchLoss = 0;

                for(int j = 0; j < batch.data.length; j++) {
                    double[] prediction = this.predict(batch.data[j]);
                    double[] losses = new double[prediction.length];

                    for(int k = 0; k < prediction.length; k++) {
                        losses[k] = batch.answers[j][k] - prediction[k];
                    }

                    batchLoss += MathUtil.meanSquareLoss(prediction, batch.answers[j]);
                    this.computeGradients(losses, batch.data[j], batchGradients, batchBiasGradients);
                }
                this.applyGradients(batchGradients, batchBiasGradients, batch.data.length,learningRate);
                epochLoss += batchLoss / batch.data.length;
            }

            if(epoch % 10 == 0) {
                System.out.println("Epoch:" + epoch + "   Loss:" + (epochLoss / batches.size()));
            }
        }
    }

    public void train(int epochs, int bSize, double learningRate) {
        int tmp = 0;
        for(int epoch = 0; epoch < epochs; epoch++) {
            ArrayList<Data> batch = loadNextBatche(bSize, tmp);
            tmp += bSize;
            double eLoss = 0;
            double bLoss = 0;

            double[][][] wGradients = new double[layers.length][][];
            double[][] bGradients = new double[layers.length][];

            for (int i = 0; i < layers.length; i++) {
                wGradients[i] = new double[layers[i].length][];
                bGradients[i] = new double[layers[i].length];
                for (int j = 0; j < layers[i].length; j++) {
                    wGradients[i][j] = new double[layers[i][j].getWeigth().length];
                }
            }

            for(Data d : batch) {
                double[] prediction = this.predict(d.getData());
                double[] loss = new double[prediction.length];

                for(int k = 0; k < prediction.length; k++) {
                    loss[k] = d.getAnswers()[k] - prediction[k];
                }

                bLoss += MathUtil.meanSquareLoss(prediction, d.getAnswers());
                this.computeGradients(loss, d.getData(), wGradients, bGradients);
                eLoss += bLoss / d.getData().length;
            }

            this.applyGradients(wGradients, bGradients, batch.size(), learningRate);

            if(epoch % 10 == 0) {
                System.out.println("Epoch:" + epoch + "   Loss:" + (eLoss / batch.size()));
            }
        }
    }

    private void computeGradients(double[] losses, double[] in, double[][][] wGradients, double[][] bGradients) {
        double[][] errors = new double[layers.length][];
        for (int i = 0; i < layers.length; i++) {
            errors[i] = new double[layers[i].length];
        }

        for (int i = layers.length - 1; i >= 0; i--) {
            for (int j = 0; j < layers[i].length; j++) {
                if (i == layers.length - 1) {
                    errors[i][j] = losses[j] * layers[i][j].derivedOutput(i == 0 ? in : values[i - 1]);
                } else{
                    double errorSum = 0;
                    for (int k = 0; k < layers[i+1].length; k++) {
                        errorSum += layers[i+1][k].getWeigth()[j] * errors[i+1][k];
                    }
                    errors[i][j] = errorSum * layers[i][j].derivedOutput(i == 0 ? in : values[i-1]);
                }

                wGradients(i, j, in, wGradients, errors);
                bGradients(i, j, bGradients, errors);
            }
        }
    }

    private void wGradients(int l, int n, double[] in, double[][][] wGradients, double[][] errors){
        for (int w = 0; w < layers[l][n].getWeigth().length; w++) {
            double input = (l == 0) ? in[w] : values[l-1][w];
            wGradients[l][n][w] += errors[l][n] * input;
        }
    }

    private void bGradients(int l, int n, double[][] bGradients, double[][] errors){
        bGradients[l][n] += errors[l][n];
    }

    private void applyGradients(double[][][] wGradients, double[][] bGradients, int bSize, double learningRate) {
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                double[] newWGradients = new double[layers[i][j].getWeigth().length];
                for (int k = 0; k < layers[i][j].getWeigth().length; k++) {
                    newWGradients[k] += (learningRate * wGradients[i][j][k]) / bSize;
                }
                double newBGradient = (learningRate * bGradients[i][j]) / bSize;
                layers[i][j].adjust(newWGradients, newBGradient);
            }
        }
    }

    public Neuron[][] getLayers() {
        return layers;
    }

    private record BatchData(double[][] data, double[][] answers) {}
}