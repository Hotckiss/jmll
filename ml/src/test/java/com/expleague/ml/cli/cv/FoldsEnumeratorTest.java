package com.expleague.ml.cli.cv;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.GridTest;
import com.expleague.ml.data.tools.FakePool;
import com.expleague.ml.data.tools.Pool;

public class FoldsEnumeratorTest extends GridTest {
  public void testEnumerate() throws Exception {
    final FakePool pool = FakePool.create(
        new VecBasedMx(1, new ArrayVec(1, 2, 3)),
        new IntSeq(1, 2, 3)
    );

    final FoldsEnumerator foldsEnumerator = new FoldsEnumerator(pool, new DetermRandom(), 3);

    assertTrue(foldsEnumerator.hasNext());
    validateFolds(foldsEnumerator.next(), pool.size(), new int[]{2, 3}, new int[]{1});

    assertTrue(foldsEnumerator.hasNext());
    validateFolds(foldsEnumerator.next(), pool.size(), new int[]{1, 3}, new int[]{2});

    assertTrue(foldsEnumerator.hasNext());
    validateFolds(foldsEnumerator.next(), pool.size(), new int[]{1, 2}, new int[]{3});

    assertFalse(foldsEnumerator.hasNext());
  }

  private static void validateFolds(final Pair<? extends Pool, ? extends Pool> pair, final int poolSize, final int[] learnTargetValues, final int[] testTargetValues) {
    final Pool learnPool = pair.getFirst();
    final Pool testPool = pair.getSecond();
    assertEquals(poolSize, learnPool.size() + testPool.size());

    final Seq learnTarget = learnPool.target(0);
    assertEquals(learnTargetValues.length, learnTarget.length());
    for (int learnTargetValue : learnTargetValues) {
      assertTrue(SeqTools.indexOf(learnTarget, learnTargetValue) != -1);

    }
    final Seq testTarget = testPool.target(0);
    assertEquals(testTargetValues.length, testTarget.length());
    for (int testTargetValue : testTargetValues) {
      assertTrue(SeqTools.indexOf(testTarget, testTargetValue) != -1);
    }
  }

  private static class DetermRandom extends FastRandom {
    private int nextResult;

    public DetermRandom() {
      super(100500);
      this.nextResult = 0;
    }

    @Override
    public int nextSimple(final Vec row) {
      return nextResult++ % row.length();
    }
  }
}