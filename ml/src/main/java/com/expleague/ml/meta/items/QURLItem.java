package com.expleague.ml.meta.items;

import com.expleague.ml.meta.DSItem;

/**
 * User: solar
 * Date: 11.07.14
 * Time: 21:25
 */
public class QURLItem extends FakeItem {
  public int queryId;
  public String url;
  public int groupId;

  public QURLItem() {
  }

  public QURLItem(final int queryId, final String url, final int groupId) {
    this.queryId = queryId;
    this.url = url;
    this.groupId = groupId;
  }

  @Override
  public String id() {
    return url + "@" + queryId;
  }
}
