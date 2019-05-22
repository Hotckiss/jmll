package com.expleague.ml.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.BFGrid;
import com.expleague.ml.GridUtils;

import java.util.Arrays;

/**
 * User: solar
 * Date: 09.11.12
 * Time: 17:56
 */
public class BFGridImpl implements BFGrid {
  final BFRowImpl[] rows;
  final BinaryFeatureImpl[] features;
  final int bfCount;
  final BFRowImpl leastNonEmptyRow;

  public BFGridImpl(final BFRowImpl[] rows) {
    this.rows = rows;
    for (final BFRowImpl row : rows) {
      row.setOwner(this);
    }
    final BFRowImpl lastRow = rows[rows.length - 1];
    bfCount = lastRow.bfStart + lastRow.borders.length;
    features = new BinaryFeatureImpl[bfCount];
    int rowIndex = 0;
    for (int i = 0; i < features.length; i++) {
      while (rowIndex < rows.length && i >= rows[rowIndex].bfEnd)
        rowIndex++;
      //System.out.println("INDEX: " + (i - rows[rowIndex].bfStart));
      features[i] = rows[rowIndex].bf(i - rows[rowIndex].bfStart);
    }

    BFRowImpl leastNonEmptyRow = null;
    for (int i = 0; i < rows.length; i++) {
      if (rows[i].size() > 0) {
        leastNonEmptyRow = leastNonEmptyRow != null ? (leastNonEmptyRow.size() > rows[i].size() ? rows[i] : leastNonEmptyRow) : rows[i];
      }
    }
    this.leastNonEmptyRow = leastNonEmptyRow;
  }

  @Override
  public BFRowImpl row(final int feature) {
    return feature < rows.length ? rows[feature] : new BFRowImpl(this, bfCount, feature, new double[0]);
  }

  @Override
  public BinaryFeatureImpl bf(final int bfIndex) {
    return features[bfIndex];
  }

  @Override
  public void binarizeTo(final Vec x, final byte[] folds) {
    for (int i = 0; i < x.dim(); i++) {
      folds[i] = (byte) rows[i].bin(x.get(i));
    }
  }

  @Override
  public int size() {
    return features.length;
  }

  @Override
  public int rows() {
    return rows.length;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof BFGridImpl)) return false;

    final BFGridImpl bfGrid = (BFGridImpl) o;

    return Arrays.equals(rows, bfGrid.rows);

  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(rows);
  }

  @Override
  public String toString() {
    return CONVERTER.convertTo(this).toString();
  }
}
