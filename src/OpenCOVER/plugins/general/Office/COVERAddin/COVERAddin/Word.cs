//#define VISTA

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Office.Interop.Word;
using System.Reflection;
using System.Threading;
using Microsoft.Office.Core;
using Microsoft.Office;
using System.Runtime.InteropServices;

using System.Windows.Forms;
using System.Drawing;

namespace OfficeConsole
{
  class Word : BasicOfficeControl
  {

    public Word()
    {
      //initMinutes();
    }

    public Microsoft.Office.Interop.Word.Application word = null;

    private string template = "";
    private _Document document = null;

    private int currentPage = -1;
    private int currentLine = -1;
    private int currentPagePollCounter = 0;

    private static object optional = Missing.Value;
    
    //public struct Minutes
    //{
    //  public string title;
    //  public string location;
    //  public string participants;
    //  public string item;
    //  public string snapshots;
    //}

    //public Minutes myMinutes;

    // WORD ///////////////////////////////////////////////////////////////////////////////////////////

    public void update() 
    { 
      if (document == null) return;

      // Limit current position polling frequency, as this is quite time demanding and hogs the CPU
      if (++this.currentPagePollCounter > 99)
      {

        int page;
        int line;

        if (getCurrentLine(out page, out line))
        {
          if (this.currentPage != page || this.currentLine != line)
          {
            this.currentPage = page;
            this.currentLine = line;
            OfficeConsole.The.sendEvent("Word.ScrollEvent " + page + "/" + line);
          }
        }

        this.currentPagePollCounter = 0;

      }

    }

    private bool getCurrentLine(out int page, out int line)
    {
      if (this.document == null)
      {
        page = 0;
        line = 0;
        return false;
      }


      /*   IntPtr handle = Native.FindWindowExW(new IntPtr(0), new IntPtr(0), "OpusApp", this.document.ActiveWindow.Caption + " - Microsoft Word");
         handle = Native.FindWindowExW(handle, new IntPtr(0), "_WwF", "");
         handle = Native.FindWindowExW(handle, new IntPtr(0), "_WwB", null);
         handle = Native.FindWindowExW(handle, new IntPtr(0), "_WwG", null);

         Native.tagRECT t = new Native.tagRECT();
         Native.GetWindowRect(handle, ref t);

         Range range = (Range)word.ActiveWindow.RangeFromPoint(t.left + 10, t.top + 10);*/
      Range range = word.Selection.Range;

      page = (int)range.get_Information(WdInformation.wdActiveEndPageNumber);
      line = (int)range.get_Information(WdInformation.wdFirstCharacterLineNumber);

      return true;
    }

    public void start()
    {

      if (this.word == null)
      {
        this.word = new Microsoft.Office.Interop.Word.Application();

        this.word.DocumentOpen += new ApplicationEvents4_DocumentOpenEventHandler(documentOpen);
        this.word.WindowActivate += new ApplicationEvents4_WindowActivateEventHandler(windowActivate);

        this.word.Visible = true;

      }
    }

    void windowActivate(Document Doc, Window Wn)
    {
      OfficeConsole.The.sendEvent("Word.WindowActivateEvent " + Wn.Caption + "/" + Doc.Name);
      this.document = Doc;
    }

    void documentOpen(Document Doc)
    {
      OfficeConsole.The.sendEvent("Word.DocumentOpenEvent " + Doc.Name);
      this.document = Doc;
    }


    //public void initMinutes()
    //{

    //myMinutes.title = "";
    //myMinutes.location = "";
    //myMinutes.participants = "";
    //myMinutes.item = "";
    //myMinutes.snapshots = "";

    //Test
    //myMinutes.title = "CoSpaces: Co-located Automotive Scenario";
    //myMinutes.location = "HLRS Stuttgart";
    //myMinutes.participants = "Mario Baalcke" + "\n" + "Andreas Kopecki";
    //myMinutes.item = "Casting factory";
    //try
    //{
    //   myMinutes.snapshots = "C:\\Devel\\snapshot1.PNG" + ";" + "C:\\Devel\\snapshot2.PNG";
    //}
    //catch
    //{
    //   myMinutes.snapshots = "";
    //}
    //}

    #region DocumentParts
    public enum DocumentPartType { String, Picture };
    public class DocumentPart
    {

      public DocumentPart(string name, string value) : this(name, value, DocumentPartType.String) { }

      public DocumentPart(string name, string value, DocumentPartType type)
      {
        PartName = name;
        PartValue = value;
        PartType = type;
      }

      private string partName;

      public string PartName
      {
        get { return partName; }
        set { partName = value; }
      }

      private string partValue;

      public string PartValue
      {
        get { return partValue; }
        set { partValue = value; }
      }

      private DocumentPartType partType;

      public DocumentPartType PartType
      {
        get { return partType; }
        set { partType = value; }
      }

    }

    private Dictionary<string, List<DocumentPart>> documentParts = new Dictionary<string, List<DocumentPart>>();

    /// <summary>
    /// Add a part to the list of parts
    /// </summary>
    /// <param name="part">the document part</param>
    public void addPart(DocumentPart part)
    {
      if (!this.documentParts.ContainsKey(part.PartName))
      {
        this.documentParts[part.PartName] = new List<DocumentPart>();
      }

      this.documentParts[part.PartName].Add(part);
    }

    /// <summary>
    /// Removes all parts with a certain value
    /// </summary>
    /// <param name="part">the document part</param>
    public void removePart(DocumentPart part)
    {

      if (!this.documentParts.ContainsKey(part.PartName)) return;

      DocumentPart target = new DocumentPart(part.PartName, part.PartValue);

      List<DocumentPart> parts = this.documentParts[part.PartName].FindAll(target.Equals);

      foreach (DocumentPart partToRemove in parts)
        this.documentParts[part.PartName].Remove(partToRemove);

      if (this.documentParts[part.PartName].Count == 0)
        this.documentParts.Remove(part.PartName);
    }


    public void addImage(string path)
    {
        word.Selection.EndKey(WdUnits.wdStory);

        if (path.Length != 0)
        {
            object tr = Microsoft.Office.Core.MsoTriState.msoTrue;
            //object fa = Microsoft.Office.Core.MsoTriState.msoFalse;
                try
                {
                        InlineShape picture = word.Selection.InlineShapes.AddPicture(path, ref tr, ref tr, ref optional);
                        picture.
                }
                catch (Exception e)
                {
                    word.Selection.TypeText(e.Message + ": " + path + "\n");
                }
                word.Selection.TypeText("\nBildunterschrift\n");
        }

    }
    /// <summary>
    /// Removes all parts and sets given part the only part.
    /// </summary>
    /// <param name="part">the document part</param>
    public void setPart(DocumentPart part)
    {

      if (this.documentParts.ContainsKey(part.PartName))
        this.documentParts[part.PartName].Clear();
      addPart(part.PartName, part.PartValue);
    }

    /// <summary>
    /// Add a part to the list of parts
    /// </summary>
    /// <param name="partName">name of the part</param>
    /// <param name="partValue">string value of the part</param>
    public void addPart(string partName, string partValue)
    {
      addPart(new DocumentPart(partName, partValue));
    }

    /// <summary>
    /// Removes a string part with a certain value
    /// </summary>
    /// <param name="partName">name of the part</param>
    /// <param name="partValue">string value of the part</param>
    public void removePart(string partName, string partValue)
    {
      removePart(new DocumentPart(partName, partValue));
    }

    /// <summary>
    /// Removes all parts and sets the parameters as the only part.
    /// </summary>
    /// <param name="partName">name of the part</param>
    /// <param name="partValue">string value of the part</param>
    public void setPart(string partName, string partValue)
    {
      setPart(new DocumentPart(partName, partValue));
    }


    //public void addSnapshot(string snap)
    //{
    //  myMinutes.snapshots = myMinutes.snapshots + ";" + snap;
    //}

    //public void addParticipant(string part)
    //{
    //  myMinutes.participants = myMinutes.participants + "\n" + part;
    //}

    private void addPartsToDocument(DocumentPart[] parts)
    {
      foreach (DocumentPart part in parts)
      {
        switch (part.PartType)
        {
          case DocumentPartType.String:
            addStringToDocument(part);
            break;
          case DocumentPartType.Picture:
            addPictureToDocument(part);
            break;
        }
      }
    }

    private void addStringToDocument(DocumentPart part)
    {
      FormFields fields = this.document.FormFields;
      foreach (FormField field in fields)
      {
        if (field.Name == part.PartName)
        {
          field.Range.Text += "\n" + part.PartValue;
        }
      }
    }


    private void addPictureToDocument(DocumentPart part)
    {
      FormFields fields = this.document.FormFields;
      foreach (FormField field in fields)
      {
        if (field.Name == part.PartName)
        {
          object msoTrue = MsoTriState.msoTrue;
          object missing = Missing.Value;
          field.Range.InlineShapes.AddPicture(part.PartValue, ref msoTrue, ref msoTrue, ref missing);
          field.Range.InsertAfter("\n");
        }
      }

    }

    #endregion

    public string Template
    {
      get { return this.template; }
      set { this.template = value; }
    }


    public void createMinutes()
    {
      if (word == null) start();

      try
      {

        if (this.document == null)
        {

          if (this.Template == "")
          {
            //System.Windows.Forms.MessageBox.Show("Minutes template not set, setting to C:\\Devel\\test.dotx", "Warning", System.Windows.Forms.MessageBoxButtons.OK, System.Windows.Forms.MessageBoxIcon.Warning);
            //this.Template = "C:\\Devel\\test.dotx";
            this.Template = "C:\\tmp\\myMeeting.dot"; // quick demo hack

          }

          object newTemplate = Missing.Value;	//Not creating a template.
          object documentType = Missing.Value;	//Plain old text document.
          object visible = true;		//Show the doc while we work.
          object template = this.Template;

          this.document =
            word.Documents.Add(ref template,
                                  ref newTemplate,
                                  ref documentType,
                                  ref visible);
        }

        // Begin quick demo hack

        object part = "title";
        this.document.Bookmarks.get_Item(ref part).Range.Text = "Cabin Air Condition";
        part = "location";
        this.document.Bookmarks.get_Item(ref part).Range.Text = "HLRS";

        part = "participants";
        List<DocumentPart> participants = this.documentParts["participants"];
        String pString = "";
        foreach (DocumentPart p in participants)
          pString += p.PartValue + "\n";
        this.document.Bookmarks.get_Item(ref part).Range.Text = pString;

        part = "item";
        this.document.Bookmarks.get_Item(ref part).Range.Text = "Assessing cabin temperatures";

        object optional = Missing.Value;
        object gotoItem = WdGoToItem.wdGoToSection;
        object gotoDirection = WdGoToDirection.wdGoToAbsolute;
        object count = 2;
        word.Selection.GoTo(ref gotoItem, ref gotoDirection, ref count, ref optional);

        List<DocumentPart> snapshots = this.documentParts["snapshot"];
        string path = "";
        foreach (DocumentPart s in snapshots)
          path += s.PartValue + ";";

        if (path.Length != 0)
        {
          string sign = ";";
          char[] delims = sign.ToCharArray();
          string[] pathlist = path.Split(delims);
          object tr = Microsoft.Office.Core.MsoTriState.msoTrue;
          //object fa = Microsoft.Office.Core.MsoTriState.msoFalse;
          foreach (string s in pathlist)
          {
            try
            {
              if (s != "")
                word.Selection.InlineShapes.AddPicture(s, ref tr, ref tr, ref optional);
            }
            catch (Exception e)
            {
              word.Selection.TypeText(e.Message + ": " + s + "\n");
            }
            word.Selection.TypeText("\n");
          }
        }


        // End quick demo hack


        string date = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm");
        string filename = "C:\\Devel\\minutes" + date + ".doc";
        save(filename);
      }
      catch { }
    }


    public bool load(string str)
    {

      if (this.word == null) return false;

      //Open an existing document.  Both the DocumentOpen and
      //DocumentChange events will fire.
      object fileName = str; //Environment.CurrentDirectory + "\\example4";
      object optional = Missing.Value;
      object visible = true;
      _Document doc;
      try
      {
        doc = word.Documents.Open(ref fileName,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref visible,
                                   ref optional,
                                   ref optional,
                                   ref optional,
                                   ref optional);
        return true;
      }
      catch
      {
        //Console.Write("{0} open failed", fileName);
        //MessageBox.Show("Failed to open Document " + fileName + " !\n Open new Document",
        //  "DocumentOpen event", MessageBoxButtons.OK, MessageBoxIcon.Information);
        //Thread.Sleep(5000);

        object template = Missing.Value;	//No template.
        object newTemplate = Missing.Value;	//Not creating a template.
        object documentType = Missing.Value;	//Plain old text document.
        object vis = true;		//Show the doc while we work.
        doc = word.Documents.Add(ref template,
                  ref newTemplate,
                  ref documentType,
                  ref vis);

        return false;

      }

    }


    public bool save(string str)
    {
      if (this.word == null) return false;

      //Save the file, use default values except for filename.  The
      //DocumentBeforeSave event will fire.
      object fileName = str; //+= "_new";

      try
      {
        _Document doc = word.ActiveDocument;
        object optional = Missing.Value;

        doc.SaveAs(ref fileName,
               ref optional, ref optional, ref optional,
               ref optional, ref optional, ref optional,
               ref optional, ref optional, ref optional,
               ref optional, ref optional, ref optional,
               ref optional, ref optional, ref optional);
        return true;
      }
      catch
      {
        return false;
      }

    }

    
    public void quit()
    {

      if (this.word == null) return;

      try
      {
        //Now use the Quit method to cleanup.
        //The Quit event will fire because saveChanges is set to wdDoNotSaveChanges.
        //If saveChanges were changed to wdSaveChanges, the Quit event will not fire.
        object saveChanges = WdSaveOptions.wdDoNotSaveChanges;
        object originalFormat = Missing.Value;
        object routeDocument = Missing.Value;
        word.Quit(ref saveChanges, ref originalFormat, ref routeDocument);
        Thread.Sleep(2000);
      }
      catch { }
      this.word = null;
    }


    public bool gotoPage(int page)
    {
      return gotoPage(page, 1);
    }

    public bool gotoPage(int page, int line)
    {

      if (this.document == null) return false;

      object goToItem = WdGoToItem.wdGoToPage;
      object goToDirection = WdGoToDirection.wdGoToAbsolute;
      object goTo = (object) page;

      Range rangeDocument = this.document.Range(ref optional, ref optional);
      Range rangePageBegin = rangeDocument.GoTo(ref goToItem, ref goToDirection, ref goTo, ref optional);

      goToDirection = WdGoToDirection.wdGoToRelative;
      goToItem = WdGoToItem.wdGoToLine;
      goTo = (object) line;

      Range rangeLine = rangePageBegin.GoTo(ref goToItem, ref goToDirection, ref goTo, ref optional);

      this.document.ActiveWindow.ScrollIntoView(rangeLine, ref optional);

      int cPage;
      int cLine;
      if (getCurrentLine(out cPage, out cLine))
      {
        if (this.currentPage != cPage || this.currentLine != cLine)
        {
          this.currentPage = cPage;
          this.currentLine = cLine;
        }
      }

      return true;

    }

  }
}