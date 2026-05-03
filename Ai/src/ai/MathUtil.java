package ai;

public class MathUtil {
    public static double sigmoid(double in){
        return 1 / (1 + Math.exp(-in));
    }

    public static double deriv(double in){
        return in * (1- in);
    }

    public static double meanSquareLoss(double[] ans, double[] correct){
        double sumSquare = 0;

        for (int i = 0; i < ans.length; i++){
            sumSquare += (correct[i] - ans[i]) * (correct[i] - ans[i]);
        }

        return sumSquare /ans.length;
    }
}