package com.expleague.ml.methods.seq.param;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

public class BettaMxParametrization implements BettaParametrization {
  private final double addToDiag;

  public BettaMxParametrization(double addToDiag) {
    this.addToDiag = addToDiag;
  }

  @Override
  public int paramCount(int stateCount) {
    return stateCount * stateCount;
  }

  @Override
  public Mx getBettaMx(Vec params, int c, int stateCount) {
    int bettaSize = paramCount(stateCount);
    Vec paramsSub = new ArrayVec(bettaSize);
    VecTools.assign(paramsSub, params.sub(c * bettaSize, (c + 1) * bettaSize));
    Mx betta = new VecBasedMx(stateCount, paramsSub);
    for (int i = 0; i < stateCount; i++) {
      betta.adjust(i, i, addToDiag);
    }
    return betta;
  }

  @Override
  public void gradientTo(Vec params, Mx dBetta, Vec dParam, int c, int stateCount) {
    VecTools.append(dParam, dBetta);
  }
}
