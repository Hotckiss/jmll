package com.expleague.ml.binarization.algorithms;

import com.expleague.commons.FileTestCase;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BFGrid;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.testUtils.TestResourceLoader;
import org.junit.Test;

import java.io.IOException;

public class EqualFrequencyBinarizationTest extends FileTestCase {
    public static Pool<?> learn;
    private static FastRandom rng = new FastRandom(0);

    private static synchronized void loadDataSet() {
        try {
            if (learn == null) {
                learn = TestResourceLoader.loadPool("features.txt.gz");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testEW1() {
        BFGrid grid = EqualFrequencyBinarization.equalFreqGrid(learn.vecData(), 2, new BuildProgressHandler(null));
    }

    @Test
    public void testEW2() {
        BFGrid grid = EqualFrequencyBinarization.equalFreqGrid(learn.vecData(), 32, new BuildProgressHandler(null));
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        loadDataSet();
    }

    @Override
    protected String getInputFileExtension() {
        return ".txt";
    }

    @Override
    protected String getResultFileExtension() {
        return ".txt";
    }

    @Override
    protected String getTestDataPath() {
        try {
            return TestResourceLoader.getFullPath("grid/");
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}