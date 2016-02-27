using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace OfficeConsole
{
  public class InternetExplorer : BasicOfficeControl
  {
    public SHDocVw.InternetExplorer ie = null;

    // TEST ///////////////////////////////////////////////////////////////////////////////////////////

    public void start()
    {
      if (this.ie == null)
      {
        object vPost, vHeaders, vFlags, vTargetFrame, vUrl;

        vFlags = null;
        vTargetFrame = null;
        vPost = null;

        vUrl = "http://google.de";

        vHeaders = null; // "Content-Type: application/x-www-form-urlencoded" + Convert.ToChar(10) + Convert.ToChar(13);

        //Create an instance of Internet Explorer and make it visible.
        this.ie = new SHDocVw.InternetExplorer();
        this.ie.Visible = true;
        this.ie.Navigate2(ref vUrl, ref vFlags, ref vTargetFrame, ref vPost, ref vHeaders);

        Thread.Sleep(5000);

      }
    }

    public void update() { }

    public bool navigate2(string url)
    {

      if (this.ie == null) return false;

      object vPost, vHeaders, vFlags, vTargetFrame, vUrl;

      vFlags = null;
      vTargetFrame = null;
      vPost = null;
      vHeaders = null;
      vUrl = url;
      try
      {
        this.ie.Navigate2(ref vUrl, ref vFlags, ref vTargetFrame, ref vPost, ref vHeaders);
        return true;
      }
      catch 
      {
        return false;
      }

      //Thread.Sleep(3000);
    }

    public void forward()
    {
      if (this.ie == null) return;

      try
      {
        this.ie.GoForward();
        Thread.Sleep(3000);
      }
      catch { }
    }

    public void back()
    {
      if (this.ie == null) return;
      try
      {
        this.ie.GoBack();
        Thread.Sleep(3000);
      }
      catch { }
    }

    public void quit()
    {
      if (ie == null) return;
      try
      {
        this.ie.Quit();
      }
      catch { }
      this.ie = null;
    }

    public bool load(string file)
    {
      return navigate2(file);
    }

    public bool save(string file)
    {
      return false;
    }
  }
}
