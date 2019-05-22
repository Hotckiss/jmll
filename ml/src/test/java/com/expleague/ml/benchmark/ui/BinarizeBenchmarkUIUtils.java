package com.expleague.ml.benchmark.ui;

import com.expleague.ml.GridTools;
import javafx.geometry.HPos;
import javafx.geometry.Insets;
import javafx.scene.chart.*;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextField;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.RowConstraints;

public class BinarizeBenchmarkUIUtils {
    public static GridPane makeGridPane() {
        GridPane gridpane = new GridPane();
        gridpane.setPadding(new Insets(5));
        gridpane.setHgap(16);
        gridpane.setVgap(7);

        RowConstraints row0 = new RowConstraints(50);
        RowConstraints row1 = new RowConstraints(50);
        RowConstraints row2 = new RowConstraints(50);
        RowConstraints row3 = new RowConstraints(300);
        ColumnConstraints column1 = new ColumnConstraints(100);
        ColumnConstraints column2 = new ColumnConstraints(100);
        ColumnConstraints column3 = new ColumnConstraints(100);
        ColumnConstraints column4 = new ColumnConstraints(100);
        ColumnConstraints column5 = new ColumnConstraints(100);
        ColumnConstraints column6 = new ColumnConstraints(100);
        ColumnConstraints column7 = new ColumnConstraints(100);
        ColumnConstraints column8 = new ColumnConstraints(100);
        ColumnConstraints column9 = new ColumnConstraints(100);
        ColumnConstraints column10 = new ColumnConstraints(100);
        ColumnConstraints column11= new ColumnConstraints(100);
        ColumnConstraints column12 = new ColumnConstraints(100);
        ColumnConstraints column13= new ColumnConstraints(100);
        ColumnConstraints column14 = new ColumnConstraints(100);
        ColumnConstraints column15 = new ColumnConstraints(100);
        ColumnConstraints column16 = new ColumnConstraints(100);
        column2.setHgrow(Priority.ALWAYS);
        gridpane.getColumnConstraints().addAll(column1, column2, column3, column4, column5, column6, column7, column8,
                column9,column10,column11,column12,column13,column14,column15,column16);
        //gridpane.getRowConstraints().addAll(row0, row1, row2, row3);

        return gridpane;
    }

    public static void addDatasetInput(GridPane gridPane, Label label, TextField field) {
        GridPane.setHalignment(label, HPos.LEFT);
        gridPane.add(label, 0, 0);
        GridPane.setHalignment(field, HPos.LEFT);
        gridPane.add(field, 1, 0, 3, 1);
    }

    public static void addFeaturesInput(GridPane gridPane, Label label, TextField field) {
        GridPane.setHalignment(label, HPos.LEFT);
        gridPane.add(label, 0, 1);
        GridPane.setHalignment(field, HPos.LEFT);
        gridPane.add(field, 1, 1, 3, 1);
    }

    public static void addTargetInput(GridPane gridPane, Label label, TextField field) {
        GridPane.setHalignment(label, HPos.LEFT);
        gridPane.add(label, 0, 2);
        GridPane.setHalignment(field, HPos.LEFT);
        gridPane.add(field, 1, 2, 3, 1);
    }

    public static void addInputs(GridPane gridpane) {
        Label datasetLabel = new Label("Dataset path:");
        TextField datasetInput = new TextField();
        BinarizeBenchmarkUIUtils.addDatasetInput(gridpane, datasetLabel, datasetInput);

        Label featuresLabel = new Label("Features to extract:");
        TextField featuresInput = new TextField();
        BinarizeBenchmarkUIUtils.addFeaturesInput(gridpane, featuresLabel, featuresInput);

        Label targetColumnLabel = new Label("Target column:");
        TextField targetInput = new TextField();
        BinarizeBenchmarkUIUtils.addTargetInput(gridpane, targetColumnLabel, targetInput);
    }

    public static void addAlgorithmChart(GridPane gridPane, XYChart.Series series, XYChart.Series seriesT, int index) {
        final NumberAxis xAxis = new NumberAxis("Iteration", 0, 2000, 250);
        final NumberAxis yAxis = new NumberAxis();
        final LineChart<Number,Number> lineChart = new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Score chart");
        lineChart.setCreateSymbols(false);
        lineChart.setStyle(".chart-series-line { -fx-stroke-width: 1px; }");
        lineChart.getData().add(series);
        lineChart.getData().add(seriesT);
        gridPane.add(lineChart, index * 4, 5, 4, 1);
    }

    public static ProgressBar addProgressBar(GridPane gridpane, int index) {
        Label binLabel = new Label("Binarize progress:");
        GridPane.setHalignment(binLabel, HPos.CENTER);
        gridpane.add(binLabel, index * 4 + 1, 4);
        ProgressBar algorithmBar = new ProgressBar();
        GridPane.setHalignment(algorithmBar, HPos.LEFT);
        gridpane.add(algorithmBar, index * 4 + 2, 4);

        return algorithmBar;
    }

    public static void addBinsChartUsage(GridPane gridPane, XYChart.Series bars, int index) {
        final CategoryAxis xAxis = new CategoryAxis();
        final NumberAxis yAxis = new NumberAxis();
        final BarChart<String,Number> barChart = new BarChart<>(xAxis, yAxis);
        barChart.setTitle("Bins usage");
        barChart.setStyle(".chart-series-line { -fx-stroke-width: 1px; }");
        barChart.getData().add(bars);
        gridPane.add(barChart, index * 4, 6, 4, 1);
    }
}
