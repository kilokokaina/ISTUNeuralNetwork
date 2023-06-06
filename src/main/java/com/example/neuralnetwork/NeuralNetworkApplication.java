package com.example.neuralnetwork;

import com.example.neuralnetwork.nn.Main;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class NeuralNetworkApplication {

    public static void main(String[] args) {
        SpringApplication.run(NeuralNetworkApplication.class, args);
    }

    @Bean
    public static CommandLineRunner cmd() {
        return Main::main;
    }

}
