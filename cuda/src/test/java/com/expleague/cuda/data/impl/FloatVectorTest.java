package com.expleague.cuda.data.impl;

import org.junit.Ignore;
import org.junit.Test;

import org.junit.Assert;

import java.util.Random;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class FloatVectorTest extends Assert {

  private static final int LENGTH = 10_000;

  private static final float DELTA = 1e-9f;

  private static final Random RANDOM = new Random();

  @Test
  public void testCreate() throws Exception {
    final float[] expected = generateHostFloatData();

    final FloatVector a = new FloatVector(expected);
    final float[] actual = a.get();
    a.destroy();

    assertArrayEquals(expected, actual, DELTA);
  }

  @Test
  public void testSet() throws Exception {
    final float[] data = generateHostFloatData();

    final FloatVector a = new FloatVector(data);
    final float[] expected = generateHostFloatData();
    a.set(expected);
    final float[] actual = a.get();
    a.destroy();

    assertArrayEquals(expected, actual, DELTA);
  }

  private float[] generateHostFloatData() {
    final float[] hostData = new float[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextFloat();
    }
    return hostData;
  }

}
