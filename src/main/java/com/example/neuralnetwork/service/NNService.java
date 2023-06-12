package com.example.neuralnetwork.service;

import com.example.neuralnetwork.nn.NeuralNetwork;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Arrays;
import java.util.List;

@Slf4j
@Service
public class NNService {
    private @Value("${neural.config}") String NEURAL_CONFIG;

    public RealMatrix transformImage(InputStream file) throws IOException {
        BufferedImage image = ImageIO.read(file);

        int w = image.getWidth();
        int h = image.getHeight();
        int[] RGBs = new int[w * h];

        image.getRGB(0, 0, w, h, RGBs, 0, w);

        double[][] imageData = new double[1024][0];

        for (int i = 0; i < RGBs.length; i++) {
            int red = (RGBs[i] >> 16) & 0xff;
            int green = (RGBs[i] >> 8) & 0xff;
            int blue = RGBs[i] & 0xff;

            double grey = 0.2125 * red + 0.7154 * green + 0.0721 * blue;

            imageData[i] = new double[] { (grey / 255.0 * 0.99) + 0.01 };
        }

        return new Array2DRowRealMatrix(imageData);
    }

    public void train(NeuralNetwork Julia, InputStream file, String trainMarker) throws IOException {
        RealMatrix target = new Array2DRowRealMatrix();

        switch (trainMarker.split(" ")[0]) {
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

        var input = transformImage(file);
        Julia.train(input, target);

        FileOutputStream outputStream = new FileOutputStream(NEURAL_CONFIG);
        ObjectOutputStream objectOutput = new ObjectOutputStream(outputStream);

        objectOutput.writeObject(Julia);

        outputStream.close();
        objectOutput.close();
    }

    public String getResultMarker(RealMatrix result) {
        List<Double> resultList = Arrays.stream(result.getData()).map(item -> item[0]).toList();

        if (resultList.stream().max(Double::compare).isPresent()) {
            double max = resultList.stream().max(Double::compare).get();

            switch (resultList.indexOf(max)) {
                case 0 -> { return "МММ"; }
                case 1 -> { return "РМ"; }
                case 2 -> { return "РМЖ"; }
                case 3 -> { return "РП"; }
                case 4 -> { return "РПЖ"; }
            }
        }

        return "Something went wrong";
    }

    public NeuralNetwork getNeuralNetwork() throws Exception {
        FileInputStream inputStream = new FileInputStream(NEURAL_CONFIG);
        ObjectInputStream objectInput = new ObjectInputStream(inputStream);

        NeuralNetwork Julia = (NeuralNetwork) objectInput.readObject();

        inputStream.close();
        objectInput.close();

        return Julia;
    }
}
