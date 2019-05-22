package com.expleague.ml.benchmark.ui;

import javafx.geometry.HPos;
import javafx.geometry.Insets;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.RowConstraints;

public class BinarizeBenchmarkUIUtils {
    public static GridPane makeGridPane() {
        GridPane gridpane = new GridPane();
        gridpane.setPadding(new Insets(5));
        gridpane.setHgap(6);
        gridpane.setVgap(6);
        ColumnConstraints column1 = new ColumnConstraints(300);
        RowConstraints row0 = new RowConstraints(50);
        RowConstraints row1 = new RowConstraints(50);
        RowConstraints row2 = new RowConstraints(50);
        RowConstraints row3 = new RowConstraints(300);
        ColumnConstraints column2 = new ColumnConstraints(50, 150, 300);
        ColumnConstraints column3 = new ColumnConstraints(300);
        ColumnConstraints column4 = new ColumnConstraints(50, 150, 300);
        column2.setHgrow(Priority.ALWAYS);
        gridpane.getColumnConstraints().addAll(column1, column2, column3, column4);
        //gridpane.getRowConstraints().addAll(row0, row1, row2, row3);

        return gridpane;
    }

    public static void addDatasetInput(GridPane gridPane, Label label, TextField field) {
        // dataset
        GridPane.setHalignment(label, HPos.RIGHT);
        gridPane.add(label, 0, 0);
        GridPane.setHalignment(field, HPos.LEFT);
        gridPane.add(field, 1, 0);
    }

    public static void addFeaturesInput(GridPane gridPane, Label label, TextField field) {
        // features
        GridPane.setHalignment(label, HPos.RIGHT);
        gridPane.add(label, 0, 1);
        GridPane.setHalignment(field, HPos.LEFT);
        gridPane.add(field, 1, 1);
    }

    public static void addTargetInput(GridPane gridPane, Label label, TextField field) {
        // dataset
        GridPane.setHalignment(label, HPos.RIGHT);
        gridPane.add(label, 0, 2);
        GridPane.setHalignment(field, HPos.LEFT);
        gridPane.add(field, 1, 2);
    }
}
