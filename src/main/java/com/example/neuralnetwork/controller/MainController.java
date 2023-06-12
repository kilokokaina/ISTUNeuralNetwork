package com.example.neuralnetwork.controller;

import com.example.neuralnetwork.nn.NeuralNetwork;
import com.example.neuralnetwork.service.NNService;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.util.Objects;

@Slf4j
@Controller
public class MainController {
    private final NNService service;

    private @Value("${path.train}") String PATH_TRAIN;

    @Autowired
    public MainController(NNService service) {
        this.service = service;
    }

    @GetMapping
    public String index() {
        return "index";
    }

    @PostMapping("query")
    public @ResponseBody String process(@RequestParam(value = "inputFile") MultipartFile inputFile) throws Exception {
        RealMatrix input = service.transformImage(inputFile.getInputStream());

        NeuralNetwork Julia = service.getNeuralNetwork();
        var result = Julia.query(input);

        return service.getResultMarker(result);
    }

    @PostMapping("train")
    public @ResponseBody String train(@RequestParam(value = "trainEpoch") int trainEpoch) throws Exception {
        File[] trainDir = new File(PATH_TRAIN).listFiles();
        NeuralNetwork Julia = new NeuralNetwork(1024, 100, 5, 0.3f);

        for (int i = 0; i < trainEpoch; i++) {
            for (File trainFile : Objects.requireNonNull(trainDir)) {
                String trainMarker = trainFile.getName().split(" ")[0];
                service.train(Julia, new FileInputStream(trainFile), trainMarker);
            }
        }

        return "Train completed";
    }
}
