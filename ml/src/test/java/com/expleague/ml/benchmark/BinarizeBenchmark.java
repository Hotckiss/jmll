package com.expleague.ml.benchmark;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BuildProgressHandler;   
import com.expleague.ml.benchmark.ml.BFGridFactory;
import com.expleague.ml.benchmark.ml.MethodRunner;
import com.expleague.ml.benchmark.ml.MethodType;
import com.expleague.ml.benchmark.ui.BinarizeBenchmarkUIUtils;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.testUtils.TestResourceLoader;
import javafx.application.Application;
import javafx.geometry.HPos;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

import java.io.IOException;

public class BinarizeBenchmark extends Application {
    public static Pool<?> dataset;
    private XYChart.Series series1 = new XYChart.Series();
    private XYChart.Series series2 = new XYChart.Series();
    private XYChart.Series series3 = new XYChart.Series();
    private XYChart.Series series4 = new XYChart.Series();
    private XYChart.Series series1T = new XYChart.Series();
    private XYChart.Series series2T = new XYChart.Series();
    private XYChart.Series series3T = new XYChart.Series();
    private XYChart.Series series4T = new XYChart.Series();

    private ProgressBar algorithm1Bar = new ProgressBar();
    private ProgressBar algorithm2Bar = new ProgressBar();
    private ProgressBar algorithm3Bar = new ProgressBar();
    private ProgressBar algorithm4Bar = new ProgressBar();

    private Label algorithm1BinTime = new Label();
    private Label algorithm2BinTime = new Label();
    private Label algorithm3BinTime = new Label();
    private Label algorithm4BinTime = new Label();

    private XYChart.Series barsUsageSeries1 = new XYChart.Series();
    private XYChart.Series barsUsageSeries2 = new XYChart.Series();
    private XYChart.Series barsUsageSeries3 = new XYChart.Series();
    private XYChart.Series barsUsageSeries4 = new XYChart.Series();

    private ComboBox box1 = new ComboBox();
    private ComboBox box2 = new ComboBox();
    private ComboBox box3 = new ComboBox();
    private ComboBox box4 = new ComboBox();

    private MethodType method1 = MethodType.MEDIAN;
    private MethodType method2 = MethodType.MEDIAN;
    private MethodType method3 = MethodType.MEDIAN;
    private MethodType method4 = MethodType.MEDIAN;

    private LineChart<Number,Number> algorithm1Chart = null;
    private LineChart<Number,Number> algorithm2Chart = null;
    private LineChart<Number,Number> algorithm3Chart = null;
    private LineChart<Number,Number> algorithm4Chart = null;

    private BarChart bar1Chart = null;
    private BarChart bar2Chart = null;
    private BarChart bar3Chart = null;
    private BarChart bar4Chart = null;

    private Label algorithm1FinalScore = new Label();
    private Label algorithm2FinalScore = new Label();
    private Label algorithm3FinalScore = new Label();
    private Label algorithm4FinalScore = new Label();

    private Label algorithm1TotalBins = new Label();
    private Label algorithm2TotalBins = new Label();
    private Label algorithm3TotalBins = new Label();
    private Label algorithm4TotalBins = new Label();

    private CheckBox thread1Enabled = new CheckBox();
    private CheckBox thread2Enabled = new CheckBox();
    private CheckBox thread3Enabled = new CheckBox();
    private CheckBox thread4Enabled = new CheckBox();

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

        Label datasetLabel = new Label("Dataset path:");
        TextField datasetInput = new TextField();
        Button datasetLoadButton = new Button();
        datasetLoadButton.setText("Load");
        BinarizeBenchmarkUIUtils.addDatasetInput(gridpane, datasetLabel, datasetInput, datasetLoadButton);

        Label featuresLabel = new Label("Features columns:");
        TextField featuresInput = new TextField();
        Button featuresApplyButton = new Button();
        featuresApplyButton.setText("Apply");
        BinarizeBenchmarkUIUtils.addFeaturesInput(gridpane, featuresLabel, featuresInput, featuresApplyButton);

        Label targetColumnLabel = new Label("Target column:");
        TextField targetInput = new TextField();
        Button targetApplyButton = new Button();
        targetApplyButton.setText("Apply");
        BinarizeBenchmarkUIUtils.addTargetInput(gridpane, targetColumnLabel, targetInput, targetApplyButton);

        setupBinarizeProgress(gridpane);

        Button runButt = new Button("Run!");
        GridPane.setHalignment(runButt, HPos.RIGHT);
        gridpane.add(runButt, 1, 3);
        setupRunButton(gridpane, runButt);

        setupScoreCharts(gridpane);
        setupBinsUsageCharts(gridpane);

        BinarizeBenchmarkUIUtils.addSettingsInput(gridpane, new Button("Settings"));

        root.getChildren().add(gridpane);
        primaryStage.setScene(new Scene(root, 1920, 900));
        primaryStage.show();
    }

    private void setupScoreCharts(GridPane gridpane) {
        series1.setName("Algorithm 1");
        series2.setName("Algorithm 2");
        series3.setName("Algorithm 3");
        series4.setName("Algorithm 4");
        series1T.setName("Algorithm 1 train");
        series2T.setName("Algorithm 2 train");
        series3T.setName("Algorithm 3 train");
        series4T.setName("Algorithm 4 train");
        algorithm1Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series1, series1T, 0);
        algorithm2Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series2, series2T, 1);
        algorithm3Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series3, series3T, 2);
        algorithm4Chart = BinarizeBenchmarkUIUtils.addAlgorithmChart(gridpane, series4, series4T, 3);
        algorithm1FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 0);
        algorithm2FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 1);
        algorithm3FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 2);
        algorithm4FinalScore = BinarizeBenchmarkUIUtils.addScore(gridpane, 3);
    }

    private void setupBinsUsageCharts(GridPane gridpane) {
        barsUsageSeries1.setName("Bins usage 1");
        barsUsageSeries2.setName("Bins usage 2");
        barsUsageSeries3.setName("Bins usage 3");
        barsUsageSeries4.setName("Bins usage 4");
        bar1Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries1, 0);
        bar2Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries2, 1);
        bar3Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries3, 2);
        bar4Chart = BinarizeBenchmarkUIUtils.addBinsChartUsage(gridpane, barsUsageSeries4, 3);
        box1 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 0);
        box2 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 1);
        box3 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 2);
        box4 = BinarizeBenchmarkUIUtils.addAlgorithmSelection(gridpane, 3);
        box1.setOnAction(event -> method1 = BFGridFactory.getAlgorithmType(box1.getValue().toString()));
        box2.setOnAction(event -> method2 = BFGridFactory.getAlgorithmType(box2.getValue().toString()));
        box3.setOnAction(event -> method3 = BFGridFactory.getAlgorithmType(box3.getValue().toString()));
        box4.setOnAction(event -> method4 = BFGridFactory.getAlgorithmType(box4.getValue().toString()));
        algorithm1TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 0);
        algorithm2TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 1);
        algorithm3TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 2);
        algorithm4TotalBins = BinarizeBenchmarkUIUtils.addBinsCount(gridpane, 3);
    }

    private void setupBinarizeProgress(GridPane gridpane) {
        algorithm1Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 0);
        algorithm2Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 1);
        algorithm3Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 2);
        algorithm4Bar = BinarizeBenchmarkUIUtils.addProgressBar(gridpane, 3);

        algorithm1BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 0);
        algorithm2BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 1);
        algorithm3BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 2);
        algorithm4BinTime = BinarizeBenchmarkUIUtils.addBinarizeTime(gridpane, 3);

        thread1Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 0);
        thread2Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 1);
        thread3Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 2);
        thread4Enabled = BinarizeBenchmarkUIUtils.addAlgorithmEnabled(gridpane, 3);
    }

    private void setupRunButton(GridPane gridpane, Button runButt) {
        runButt.setOnAction(event -> {
            if (thread1Enabled.isSelected()) {
                final String name1 = BFGridFactory.getAlgorithmName(method1);
                algorithm1Chart.setTitle(name1 + " score");
                series1.setName(name1 + " validate");
                series1T.setName(name1 + " train");
                barsUsageSeries1.setName(name1 + " bins usage");
                bar1Chart.setTitle(name1 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm1Log.txt",
                                dataset,
                                new FastRandom(1),
                                method1,
                                0.2,
                                series1,
                                series1T,
                                32,
                                6,
                                2000,
                                0.005,
                                new BuildProgressHandler(algorithm1Bar, BFGridFactory.getStepsCount(method1, dataset.vecData(), 32)),
                                0,
                                barsUsageSeries1,
                                algorithm1BinTime,
                                algorithm1FinalScore,
                                algorithm1TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }

            if(thread2Enabled.isSelected()) {
                final String name2 = BFGridFactory.getAlgorithmName(method2);
                algorithm2Chart.setTitle(name2 + " score");
                series2.setName(name2 + " validate");
                series2T.setName(name2 + " train");
                barsUsageSeries2.setName(name2 + " bins usage");
                bar2Chart.setTitle(name2 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm2Log.txt",
                                dataset,
                                new FastRandom(1),
                                method2,
                                0.2,
                                series2,
                                series2T,
                                32,
                                6,
                                2000,
                                0.005,
                                new BuildProgressHandler(algorithm2Bar, BFGridFactory.getStepsCount(method2, dataset.vecData(), 32)),
                                1,
                                barsUsageSeries2,
                                algorithm2BinTime,
                                algorithm2FinalScore,
                                algorithm2TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }

            if (thread3Enabled.isSelected()) {
                final String name3 = BFGridFactory.getAlgorithmName(method3);
                algorithm3Chart.setTitle(name3 + " score");
                series3.setName(name3 + " validate");
                series3T.setName(name3 + " train");
                barsUsageSeries3.setName(name3 + " bins usage");
                bar3Chart.setTitle(name3 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm3Log.txt",
                                dataset,
                                new FastRandom(1),
                                method3,
                                0.2,
                                series3,
                                series3T,
                                32,
                                6,
                                2000,
                                0.005,
                                new BuildProgressHandler(algorithm3Bar, BFGridFactory.getStepsCount(method3, dataset.vecData(), 32)),
                                2,
                                barsUsageSeries3,
                                algorithm3BinTime, algorithm3FinalScore, algorithm3TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }

            if (thread4Enabled.isSelected()) {
                final String name4 = BFGridFactory.getAlgorithmName(method4);
                algorithm4Chart.setTitle(name4 + " score");
                series4.setName(name4 + " validate");
                series4T.setName(name4 + " train");
                barsUsageSeries4.setName(name4 + " bins usage");
                bar4Chart.setTitle(name4 + " bins usage");
                new Thread(() -> {
                    try {
                        new MethodRunner("Algorithm4Log.txt",
                                dataset,
                                new FastRandom(1),
                                method4,
                                0.2,
                                series4,
                                series4T,
                                32,
                                6,
                                2000,
                                0.005,
                                new BuildProgressHandler(algorithm4Bar, BFGridFactory.getStepsCount(method4, dataset.vecData(), 32)),
                                3,
                                barsUsageSeries4,
                                algorithm4BinTime, algorithm4FinalScore, algorithm4TotalBins).run();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();
            }
        });
    }
}
