package trainingData;

import java.util.ArrayList;

public class DataManager {
    private static ArrayList<Data> data = new ArrayList<Data>();

    public static void addData(Data data) {
        DataManager.data.add(data);
    }

    public static Data getData(int i) {
        return data.get(i);
    }

    public static int getSize(){
        return data.size();
    }

    public static void shuffle(){
        java.util.Collections.shuffle(data);
    }
}