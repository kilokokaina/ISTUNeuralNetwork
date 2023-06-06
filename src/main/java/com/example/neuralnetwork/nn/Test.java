package com.example.neuralnetwork.nn;

import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.util.Arrays;
import java.util.Objects;

@Slf4j
public class Test {
    public static void main(String[] args) throws IOException {
        File[] globalDir = new File("/Users/nikol/Desktop/Often/Учебка/ВКР/Test/Output").listFiles();
        File[] localDir = new File("src/main/resources/data/test/output").listFiles();

        log.info("Global Dir:");
        for (File file : Objects.requireNonNull(globalDir)) {
            if (!file.getName().split("")[0].equals(".")) {
                log.info(Arrays.deepToString(Main.getInput(file).getData()));
            }
        }

        log.info("\nLocal Dir:");
        for (File file : Objects.requireNonNull(localDir)) {
            if (!file.getName().split("")[0].equals(".")) {
                log.info(Arrays.deepToString(Main.getInput(file).getData()));
            }
        }
    }
}