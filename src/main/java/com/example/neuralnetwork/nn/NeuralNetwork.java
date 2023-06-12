package com.example.neuralnetwork.nn;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

@Slf4j
@Data
public class NeuralNetwork implements Serializable {
    private int inputNodes, hiddenNodes, outputNodes;
    private float learningRate;

    @Serial
    private static final long serialVersionUID = 1L;

    private RealMatrix weightInputHidden, weightHiddenOutput;

    private RealMatrix activeFunc(RealMatrix x) {
        double[][] inputData = x.getData();

        Sigmoid s = new Sigmoid();
        for (double[] value : inputData) value[0] = s.value(value[0]);

        return new Array2DRowRealMatrix(inputData);
    }

    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, float learningRate) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.learningRate = learningRate;

        Random random = new Random();
        double[][] matrixDataIH = new double[hiddenNodes][inputNodes];
        double[][] matrixDataHO = new double[outputNodes][hiddenNodes];

        for (int i = 0; i < this.hiddenNodes; i++) {
            for (int j = 0; j < this.inputNodes; j++) {
                matrixDataIH[i][j] = random.nextGaussian(0.0, Math.pow(hiddenNodes, -0.5));
            }
        }


        for (int i = 0; i < this.outputNodes; i++) {
            for (int j = 0; j < this.hiddenNodes; j++) {
                matrixDataHO[i][j] = random.nextGaussian(0.0, Math.pow(outputNodes, -0.5));
            }
        }

        weightInputHidden = MatrixUtils.createRealMatrix(matrixDataIH);
        weightHiddenOutput = MatrixUtils.createRealMatrix(matrixDataHO);
    }

    private RealMatrix dot(RealMatrix a, RealMatrix b) {
        var aData = a.getData();
        var bData = b.getData();

        double[][] resultData = new double[aData.length][];
        for (int i = 0; i < aData.length; i++) resultData[i] = new double[]{ aData[i][0] * bData[i][0] };

        return new Array2DRowRealMatrix(resultData);
    }

    public RealMatrix gradientDescent(RealMatrix errors, RealMatrix output,  RealMatrix layer) {
        double[][] unitData = new double[output.getRowDimension()][];
        for (int i = 0; i < output.getRowDimension(); i++) unitData[i] = new double[] { 1 };

        var unitVector = new Array2DRowRealMatrix(unitData);
        var errorOutputMinusOne = dot(dot(errors, output), unitVector.subtract(output));

        var matrixDot = errorOutputMinusOne.multiply(layer.transpose());

        return matrixDot.scalarMultiply(learningRate);
    }

    public void train(RealMatrix input, RealMatrix target) {
        var hiddenInput = weightInputHidden.multiply(input);
        var hiddenOutput = activeFunc(hiddenInput);

        var finalInput = weightHiddenOutput.multiply(hiddenOutput);
        var finalOutput = activeFunc(finalInput);

        var outputErrors = target.subtract(finalOutput);

        var hiddenErrors = weightHiddenOutput.transpose().multiply(outputErrors);

        var whoChange = gradientDescent(outputErrors, finalOutput, hiddenOutput);
        weightHiddenOutput = weightHiddenOutput.add(whoChange);

        var wihChange = gradientDescent(hiddenErrors, hiddenOutput, input);
        weightInputHidden = weightInputHidden.add(wihChange);
    }

    public RealMatrix query(RealMatrix input) {
        var hiddenInput = weightInputHidden.multiply(input);
        var hiddenOutput = activeFunc(hiddenInput);

        var finalInput = weightHiddenOutput.multiply(hiddenOutput);

        return activeFunc(finalInput);
    }
}
