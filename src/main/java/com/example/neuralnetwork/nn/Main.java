package com.example.neuralnetwork.nn;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

@Slf4j
public class Main {
    private final static String TRAIN_PATH = "src/main/resources/data/test/output";
    private final static String TEST_PATH = "src/main/resources/data/test/output";

    public static RealMatrix getInput(File inputFile) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String fileData = reader.readLine();
        reader.close();

        String[] data = fileData.split(" ");

        double[][] dataArray = new double[1024][1];
        for (int i = 0; i < data.length; i++) {
            dataArray[i] = new double[]{ Double.parseDouble(data[i]) / 255.0 * 0.99 + 0.01 };
        }

        return new Array2DRowRealMatrix(dataArray);
    }

    public static void test(NeuralNetwork n) throws IOException {
        float percent = 0;

        File[] listFiles = Objects.requireNonNull(new File(TEST_PATH).listFiles());
        for (File testFile : listFiles) {
            String testMarker = testFile.getName().split(" ")[0];

            var input = getInput(testFile);
            var result = n.query(input);

            List<Double> resultList = Arrays.stream(result.getData()).map(item -> item[0]).toList();
            if (resultList.stream().max(Double::compare).isPresent()) {
                double max = resultList.stream().max(Double::compare).get();

                log.info("Test: " + testMarker);

                switch (resultList.indexOf(max)) {
                    case 0 -> {
                        log.info("Answer: МММ");
                        if (testMarker.equals("МММ")) percent++;
                    }
                    case 1 -> {
                        log.info("Answer: РМ");
                        if (testMarker.equals("РМ")) percent++;
                    }
                    case 2 -> {
                        log.info("Answer: РМЖ");
                        if (testMarker.equals("РМЖ")) percent++;
                    }
                    case 3 -> {
                        log.info("Answer: РП");
                        if (testMarker.equals("РП")) percent++;
                    }
                    case 4 -> {
                        log.info("Answer: РПЖ");
                        if (testMarker.equals("РПЖ")) percent++;
                    }
                }

                log.info("Output: " + resultList + "\n");
            }
        }

        log.info("Result: " + Math.round(percent / (float) listFiles.length * 100f) + "%");
    }

    public static void main(String[] args) throws IOException {
        NeuralNetwork Julia = new NeuralNetwork(1024, 100, 5, 0.3f);
        RealMatrix target = new Array2DRowRealMatrix();

        int trainEpoch = 50;

        for (int e = 1; e <= trainEpoch; e++) {
            log.info(((float) e / (float) trainEpoch * 100) + "%");

            File trainDir = new File(TRAIN_PATH);
            for (File trainFile : Objects.requireNonNull(trainDir.listFiles())) {
                String trainMarker = trainFile.getName().split(" ")[0];

                switch (trainMarker) {
                    case "МММ" -> target = new Array2DRowRealMatrix(new double[][]{
                            {0.99}, {0.01}, {0.01}, {0.01}, {0.01}
                    });
                    case "РМ" -> target = new Array2DRowRealMatrix(new double[][]{
                            {0.01}, {0.99}, {0.01}, {0.01}, {0.01}
                    });
                    case "РМЖ" -> target = new Array2DRowRealMatrix(new double[][]{
                            {0.01}, {0.01}, {0.99}, {0.01}, {0.01}
                    });
                    case "РП" -> target = new Array2DRowRealMatrix(new double[][]{
                            {0.01}, {0.01}, {0.01}, {0.99}, {0.01}
                    });
                    case "РПЖ" -> target = new Array2DRowRealMatrix(new double[][]{
                            {0.01}, {0.01}, {0.01}, {0.01}, {0.99}
                    });
                }

                var input = getInput(trainFile);
                Julia.train(input, target);
            }
        }

        log.info("The training is completed...\n");

        Main.test(Julia);
    }
}
