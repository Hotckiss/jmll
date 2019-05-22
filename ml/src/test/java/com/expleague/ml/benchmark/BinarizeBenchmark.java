package com.expleague.ml.benchmark;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.benchmark.ml.BFGridFactory;
import com.expleague.ml.benchmark.ml.MethodRunner;
import com.expleague.ml.benchmark.ml.MethodType;
import com.expleague.ml.benchmark.ui.BinarizeBenchmarkUIUtils;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.testUtils.TestResourceLoader;
import javafx.application.Application;
import javafx.geometry.HPos;
import javafx.scene.Scene;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

import java.io.IOException;

public class BinarizeBenchmark extends Application {
    public static Pool<?> dataset;
    private XYChart.Series series1 = new XYChart.Series();
    private XYChart.Series series2 = new XYChart.Series();

    private static synchronized void loadDataSet() {
        try {
            dataset = TestResourceLoader.loadPool("features.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        loadDataSet();
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Binarization Benchmark");

        StackPane root = new StackPane();

        GridPane gridpane = BinarizeBenchmarkUIUtils.makeGridPane();
        BinarizeBenchmarkUIUtils.addInputs(gridpane);

        Label bin1Label = new Label("Binarize progress:");
        GridPane.setHalignment(bin1Label, HPos.CENTER);
        gridpane.add(bin1Label, 0, 4);
        ProgressBar algorithm1Bar = new ProgressBar();
        GridPane.setHalignment(algorithm1Bar, HPos.LEFT);
        gridpane.add(algorithm1Bar, 1, 4);

        Label bin2Label = new Label("Binarize progress:");
        GridPane.setHalignment(bin2Label, HPos.CENTER);
        gridpane.add(bin2Label, 2, 4);
        ProgressBar algorithm2Bar = new ProgressBar();
        GridPane.setHalignment(algorithm2Bar, HPos.LEFT);
        gridpane.add(algorithm2Bar, 3, 4);

        Button runButt = new Button("Run!");
        GridPane.setHalignment(runButt, HPos.RIGHT);
        gridpane.add(runButt, 1, 3);

        runButt.setOnAction(event -> {
            new Thread(() -> {
                try {
                    new MethodRunner("Algorithm1Log.txt",
                            dataset,
                            new FastRandom(1),
                            MethodType.MEDIAN,
                            0.2,
                            series1,
                            32,
                            6,
                            2000,
                            0.005,
                            new BuildProgressHandler(algorithm1Bar, BFGridFactory.getStepsCount(MethodType.MEDIAN, dataset.vecData(), 32))).run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            new Thread(() -> {
                try {
                    new MethodRunner("Algorithm2Log.txt",
                            dataset,
                            new FastRandom(1),
                            MethodType.PROBABILITY_PRESORT,
                            0.2,
                            series2,
                            32,
                            6,
                            2000,
                            0.005,
                            new BuildProgressHandler(algorithm2Bar, BFGridFactory.getStepsCount(MethodType.PROBABILITY_PRESORT, dataset.vecData(), 32))).run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        });

        series1.setName("Algorithm 1");
        series2.setName("Algorithm 2");
        BinarizeBenchmarkUIUtils.addAlgorithm1Chart(gridpane, series1);
        BinarizeBenchmarkUIUtils.addAlgorithm2Chart(gridpane, series2);

        root.getChildren().add(gridpane);
        primaryStage.setScene(new Scene(root, 800, 600));
        primaryStage.show();
    }
}
