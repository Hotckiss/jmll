package com.expleague.ml.func.generic;

import com.expleague.commons.math.TransC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Identity extends TransC1.Stub {
  private final int dim;

  public Identity(int dim) {
    this.dim = dim;
  }


  @Override
  public Vec gradientRowTo(Vec x, Vec to, int index) {
    to.set(index, 1.);
    return to;
  }

  @Override
  public Vec transTo(Vec argument, Vec to) {
    return VecTools.assign(to, argument);
  }

  @Override
  public int xdim() {
    return dim;
  }

  @Override
  public int ydim() {
    return dim;
  }
}
