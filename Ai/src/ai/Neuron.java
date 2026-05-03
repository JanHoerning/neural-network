package ai;

import java.util.Random;

public class Neuron{
    private double[] weigth;
    private double bias;

    Neuron(int in){
        weigth = new double[in];
        Random rand = new Random();
        for (int i = 0; i < weigth.length; i++) weigth[i] = rand.nextGaussian() * 0.1;
        bias = rand.nextGaussian() * 0.1;
    }

    Neuron(double[] in){
        this.weigth = new double[in.length - 1];
        System.arraycopy(in, 0, this.weigth, 0, in.length - 1);
        this.bias = in[in.length-1];
    }

    public double compute(double[] values){
        double sum = 0;
        for (int i = 0; i < weigth.length; i++) sum += weigth[i] * values[i];

        return MathUtil.sigmoid(sum + bias);
    }

    public void adjust(double[] w, double b){
        for (int i = 0; i < weigth.length; i++) weigth[i] += w[i];
        bias += b;
    }

    public double derivedOutput(double[] val){
        return MathUtil.deriv(compute(val));
    }

    public double[] getWeigth() {
        return weigth;
    }

    public double getBias() {
        return bias;
    }
}