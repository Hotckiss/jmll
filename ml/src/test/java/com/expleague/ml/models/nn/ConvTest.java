package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.models.nn.layers.ConstSizeInput3D;
import com.expleague.ml.models.nn.layers.ConvLayerBuilder;
import com.expleague.ml.models.nn.layers.Layer;
import com.expleague.ml.models.nn.layers.OneOutLayer;
import org.junit.Test;

import java.util.function.Consumer;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class ConvTest {
  private static final NeuralSpider<Vec> spider = new NeuralSpider<>();
  private static final FastRandom rng = new FastRandom();
  private static final int ROUNDS = 10;

  @Test
  public void convLayerTest() {
    final int width = 15;
    final int height = 15;
    final int channels = 3;

    for (int ksizeX = 1; ksizeX <= height; ksizeX++) {
      for (int ksizeY = 1; ksizeY <= width; ksizeY++) {
        for (int strideX = 1; strideX < height - ksizeX; strideX++) {
          for (int strideY = 1; strideY < width - ksizeY; strideY++) {
            System.out.println("kSize [" + ksizeX + ", " + ksizeY + "], " +
                "stride [" + strideX + ", " + strideY + "]");
            final NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
                new ConstSizeInput3D(height, width, channels))
                .append(ConvLayerBuilder.create()
                    .ksize(ksizeX, ksizeY)
                    .stride(strideX, strideY)
                    .channels(channels))
                .build(new OneOutLayer());

            testNetwork(network, width, height, channels);
          }
        }
      }
    }
  }

  @Test
  public void multiLayerTest() {
    final int width = 100;
    final int height = 100;
    final int channels = 3;

    for (int shot = 0; shot < 50; shot++) {
      final int numLayers = rng.nextInt(4) + 1;

      NetworkBuilder<Vec> builder = new NetworkBuilder<>(new ConstSizeInput3D(height, width, channels));

      int curWidth = width;
      int curHeight = height;
      for (int i = 0; i < numLayers; i++) {
        final int ksizeX = rng.nextPoisson(3.) + 1;
        final int ksizeY = rng.nextPoisson(3.) + 1;
        final int strideX = rng.nextInt(2) + 1;
        final int strideY = rng.nextInt(2) + 1;

        curWidth = (curWidth - ksizeY) / strideY + 1;
        curHeight = (curHeight - ksizeX) / strideX + 1;

        builder.append(ConvLayerBuilder.create()
            .ksize(ksizeX, ksizeY).stride(strideX, strideY).channels(channels));
      }

      final NetworkBuilder<Vec>.Network network = builder.build(new OneOutLayer());
      System.out.println(network);
      testNetwork(network, width, height, channels);
    }
  }

  public void testNetwork(NetworkBuilder<Vec>.Network network, int width, int height, int channels) {
    for (int i = 0; i < ROUNDS; i++) {
      final Vec weights = new ArrayVec(network.wdim());
      VecTools.fillUniform(weights, rng);

      final Vec arg = new ArrayVec(width * height * channels);
      VecTools.fillUniform(arg, rng);
      final Vec[] out = {arg};

      network.layers().forEachOrdered(
          new Consumer<Layer>() {
            int curWidth = width;
            int curHeight = height;
            int wStart = 0;
            int prevChannels = channels;

            @Override
            public void accept(Layer layer) {
              if (layer instanceof ConvLayerBuilder.ConvLayer) {
                final ConvLayerBuilder.ConvLayer convLayer = (ConvLayerBuilder.ConvLayer) layer;
                final int ksizeX = convLayer.kSizeX();
                final int ksizeY = convLayer.kSizeY();
                final int strideX = convLayer.strideX();
                final int strideY = convLayer.strideY();

                final int dstHeight = (curHeight - ksizeX) / strideX + 1;
                final int dstWidth = (curWidth - ksizeY) / strideY + 1;
                final int channels = convLayer.channels();
                final int ydim = dstHeight * dstWidth * channels;
                final int wdim = (ksizeX * ksizeY * prevChannels + 1) * channels;

                assertEquals(ydim, convLayer.ydim());
                assertEquals(wdim, convLayer.wdim());
                Vec w = weights.sub(wStart, wdim);
                out[0] = conv(out[0], w, ksizeX, ksizeY,
                    strideX, strideY, dstWidth, dstHeight,
                    curWidth, prevChannels, channels);

                curWidth = dstWidth;
                curHeight = dstHeight;
                prevChannels = channels;
                wStart += wdim;
              }
            }
          }
      );

      final Vec expect = out[0];
      final Vec result = spider.compute(network, arg, weights);
      assertEquals(expect.dim(), result.dim());

      final boolean condition = VecTools.equals(expect, result);
      if (!condition) {
        if (result.dim() != expect.dim()) {
          System.out.println("dims differ");
        } else{
          for (int j = 0; j < result.dim(); j++) {
            if (expect.get(j) != result.get(j)) {
              System.out.println("pos " + j + ": expected " + expect.get(j) + " but got " + result.get(j));
              System.out.println("result: " + result);
              break;
            }
          }
        }

      }
      assertTrue(condition);
    }
  }

  private Vec conv(Vec arg, Vec weights, int ksizeX, int ksizeY,
                   int strideX, int strideY, int dstWidth, int dstHeight,
                   int width, int inChannels, int outChannels) {
    Vec result = new ArrayVec(dstHeight * dstWidth * outChannels);
    final int weightChannelDim = ksizeX * ksizeY * inChannels;
    final int biasStart = weightChannelDim * outChannels;

    for (int x = 0; x < dstHeight; x++) {
      for (int y = 0; y < dstWidth; y++) {
        for (int c = 0; c < outChannels; c++) {
          final double con = conv(x * strideX, y * strideY, c, arg, weights,
              ksizeX, ksizeY, width, weightChannelDim, inChannels)
              + weights.get(biasStart + c);
          result.set((x * dstWidth + y) * outChannels + c, con);
        }
      }
    }

    return result;
  }

  private double conv(int x, int y, int c, Vec arg, Vec weights, int ksizeX, int ksizeY,
                      int width, int weightChannelDim, int channels) {
    double result = 0.;
    for (int i = 0; i < ksizeX; i++) {
      for (int j = 0; j < ksizeY; j++) {
        for (int k = 0; k < channels; k++) {
          final int idx = ((x + i) * width + y + j) * channels + k;
          final double a = arg.get(idx);
          final double b = weights.get(c * weightChannelDim + (i * ksizeY + j) * channels + k);
          result += a * b;
        }
      }
    }

    return result;
  }
}
