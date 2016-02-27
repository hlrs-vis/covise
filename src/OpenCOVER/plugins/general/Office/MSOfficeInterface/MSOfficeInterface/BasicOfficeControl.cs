using System;
using System.Collections.Generic;
using System.Text;

namespace OfficeConsole
{
  public interface BasicOfficeControl
  {
    void start();
    void quit();
    bool load(string url);
    bool save(string url);

    void update();
  }
}
