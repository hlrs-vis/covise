using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Xml.Linq;
using Word = Microsoft.Office.Interop.Word;
using Office = Microsoft.Office.Core;

#pragma warning disable 414
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
