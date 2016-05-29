using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

using Microsoft.Office.Interop.PowerPoint;

namespace PPTAddIn
{
  public class PowerPoint
  {
    private Application powerpoint = null;
    //private bool bAssistantOn;

    Presentations presentations = null;
    _Presentation presentation = null;
    Slides slides = null;
    //Microsoft.Office.Interop.PowerPoint._Slide objSlide = null;
    //Microsoft.Office.Interop.PowerPoint.TextRange objTextRng = null;
    //Microsoft.Office.Interop.PowerPoint.Shapes objShapes = null;
    //Microsoft.Office.Interop.PowerPoint.Shape objShape = null;
    SlideShowWindow slideShowWindow = null;
    SlideShowWindows slideShowWindows = null;
    SlideShowTransition slideShowTransition = null;
    SlideShowSettings slideShowSettings = null;
    SlideRange slideRange = null;

    // POWERPOINT /////////////////////////////////////////////////////////////////////////////////////

    public void start()
    {
      if (this.powerpoint != null) return;
      this.powerpoint = ThisAddIn.The.Application;
      if (this.powerpoint == null)
      {
          this.powerpoint = new Microsoft.Office.Interop.PowerPoint.Application();
      }
      this.powerpoint.Visible = Microsoft.Office.Core.MsoTriState.msoTrue;

      this.powerpoint.SlideShowBegin += new EApplication_SlideShowBeginEventHandler(slideShowStarted);
      this.powerpoint.SlideShowEnd += new EApplication_SlideShowEndEventHandler(slideShowStopped);
      this.powerpoint.SlideShowOnNext += new EApplication_SlideShowOnNextEventHandler(slideShowOnNext);
      this.powerpoint.SlideShowOnPrevious += new EApplication_SlideShowOnPreviousEventHandler(slideShowOnPrevious);
      this.powerpoint.SlideShowNextClick += new EApplication_SlideShowNextClickEventHandler(slideShowNextClick);
      this.powerpoint.SlideShowNextSlide += new EApplication_SlideShowNextSlideEventHandler(slideShowNextSlide);

    }

    public void update() { }

    public bool load(string file)
    {
      if (this.powerpoint == null) return false;
      try
      {
        string fileName = file.Trim();
        presentations = powerpoint.Presentations;
        presentation = presentations.Open(fileName,
                                          Microsoft.Office.Core.MsoTriState.msoFalse,
                                          Microsoft.Office.Core.MsoTriState.msoFalse,
                                          Microsoft.Office.Core.MsoTriState.msoTrue);
        slides = presentation.Slides;
        return true;
      }
      catch
      {
        presentation = null;
        slides = null;
        return false;
      }
      //Thread.Sleep(5000);
    }

    public bool startSlideShow()
    {
      if (this.powerpoint == null) return false;
      if (this.powerpoint.Presentations.Count == 0) return false;

      stopSlideShow();
      if (slides == null)
      {
          presentations = powerpoint.Presentations;
          presentation = presentations._Index(1);
          if (presentation != null)
          {
              slides = presentation.Slides;
          }
      }
      if (slides != null)
      {

          int all = slides.Count;
          int[] SlideIdx = new int[all];
          for (int i = 0; i < all; i++) SlideIdx[i] = i + 1;
          slideRange = slides.Range(SlideIdx);

          slideShowTransition = slideRange.SlideShowTransition;
          slideShowTransition.AdvanceOnTime = Microsoft.Office.Core.MsoTriState.msoFalse;
          slideShowTransition.EntryEffect = Microsoft.Office.Interop.PowerPoint.PpEntryEffect.ppEffectBoxOut;

          //Prevent Office Assistant from displaying alert messages:
          //dumps if NotFiniteNumberException installed   bAssistantOn = powerpoint.Assistant.On;
          //   powerpoint.Assistant.On = false;
          //Run the Slide show 
          slideShowSettings = presentation.SlideShowSettings;
          slideShowSettings.StartingSlide = 1;
          slideShowSettings.EndingSlide = all;
          slideShowWindow = slideShowSettings.Run();
          slideShowWindows = powerpoint.SlideShowWindows;
      }

      return true;

    }

    public bool stopSlideShow()
    {
      if (this.powerpoint == null) return false;
      if (this.powerpoint.Presentations.Count == 0) return false;



      if (slideShowWindow != null)
      {
        slideShowWindow.View.Exit();
        slideShowWindow = null;
        return true;
      }
      else
        return false;
    }

    public void next()
    {
        try
        {
            powerpoint.Presentations._Index(1).SlideShowWindow.View.Next();
        }
        catch
        {
            startSlideShow();
            powerpoint.Presentations._Index(1).SlideShowWindow.View.Next();
        }
    }
    public void previous()
    {
        try
        {
            powerpoint.Presentations._Index(1).SlideShowWindow.View.Previous();
        }
        catch
        {
            startSlideShow();
            powerpoint.Presentations._Index(1).SlideShowWindow.View.Previous();
        }
    }
    public int CurrentSlide
    {
      get
      {
        if (this.powerpoint == null) return 0;
        if (this.powerpoint.Presentations.Count == 0) return 0;
        if (this.slideShowWindow == null) return 0;

        return this.slideShowWindow.View.CurrentShowPosition;
      }

      set
      {
        if (this.powerpoint == null) return;
        if (this.powerpoint.Presentations.Count == 0) return;

        if (value > slides.Count) return;
        if (slideShowWindow != null)
        {
          slideShowWindow.View.GotoSlide(value, Microsoft.Office.Core.MsoTriState.msoTrue);
        }
      }
    }

    public void quitPresentation(Presentation presentation)
    {
      //Reenable Office Assisant, if it was on:
      if (this.powerpoint == null) return;
      if (this.powerpoint.Presentations.Count == 0) return;
      if (presentation == null) return;

      //if (this.bAssistantOn)
      //{
      //  this.powerpoint.Assistant.On = true;
      //  this.powerpoint.Assistant.Visible = false;
      //}
      presentation.Close();
      if (presentation == this.presentation) this.presentation = null;
    }

    public void quit()
    {
      if (this.powerpoint == null) return;
      foreach (Presentation presentation in this.powerpoint.Presentations) 
        quitPresentation(presentation);
      this.powerpoint.Quit();
      this.powerpoint = null;
    }

    public bool save(string file)
    {
      return false;
    }


    // ------------------- Event handlers -----------------------

    public void slideShowStarted(SlideShowWindow window)
    {
      this.slideShowWindow = window;
      ThisAddIn.The.sendStringMessage("PowerPoint.SlideShowBeginEvent " + window.Presentation.Name);
    }

    public void slideShowStopped(Presentation presentation)
    {
      this.slideShowWindow = null;
      ThisAddIn.The.sendStringMessage("PowerPoint.SlideShowEndEvent " + presentation.Name);
    }

    void slideShowOnPrevious(SlideShowWindow Wn)
    {
        ThisAddIn.The.sendStringMessage("PowerPoint.SlideShowOnPreviousEvent " + CurrentSlide);
    }

    void slideShowOnNext(SlideShowWindow Wn)
    {
        ThisAddIn.The.sendStringMessage("PowerPoint.SlideShowOnNextEvent " + CurrentSlide);
    }

    void slideShowNextClick(SlideShowWindow Wn, Effect nEffect)
    {
        ThisAddIn.The.sendStringMessage("PowerPoint.SlideShowNextClickEvent " + CurrentSlide);
    }

    void slideShowNextSlide(SlideShowWindow Wn)
    {
        ThisAddIn.The.sendStringMessage("PowerPoint.SlideShowNextSlideEvent " + CurrentSlide);
    }

  }
}
