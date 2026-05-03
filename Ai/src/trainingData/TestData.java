package trainingData;

public class TestData implements Data {
    private  double[] data, answers;
    public TestData(double[] data, double[] answers){
        this.data = data;
        this.answers = answers;
    }

    @Override
    public double[] getData() {
        return this.data;
    }

    @Override
    public double[] getAnswers() {
        return this.answers;
    }
}