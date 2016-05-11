using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using System.Threading;
using Word = Microsoft.Office.Interop.Word;
using Office = Microsoft.Office.Core;
using Microsoft.Office.Tools.Word;
using OpenCOVERInterface;
using COVERAddin;
using System.Reflection;
using Microsoft.Office.Interop.Word;

namespace COVERAddin
{

    public partial class ThisAddIn
    {
        Word word;
    private static ThisAddIn coverAddin = null;
    public static ThisAddIn The
    {
      get { return coverAddin; }
    }
        Thread coverThread;
        private static object optional = Missing.Value;
        private void DocumentBeforeDoubleClick()
        {
            Microsoft.Office.Tools.Word.Document vstoDoc = Globals.Factory.GetVstoObject(this.Application.ActiveDocument);
            vstoDoc.BeforeDoubleClick += new Microsoft.Office.Tools.Word.ClickEventHandler(ThisDocument_BeforeDoubleClick);
        }

        void ThisDocument_BeforeDoubleClick(object sender, Microsoft.Office.Tools.Word.ClickEventArgs e)
        {
            Microsoft.Office.Tools.Word.Document vstoDoc = Globals.Factory.GetVstoObject(this.Application.ActiveDocument);

            int numShapes = Application.Selection.InlineShapes.Count;
            if (numShapes > 0)
            {
                InlineShape s = Application.Selection.InlineShapes[1];
                sendViewpointMessage(s.AlternativeText);
            }
        }
        public void sendApplicationType()
        {

            MessageBuffer mb = new MessageBuffer();
            mb.add("Word");
            mb.add("Microsoft Word Add-in");

            COVER.Instance.sendMessage(mb.buf, COVER.MessageTypes.ApplicationType);
        }
        public void messageLoop()
        {
            while (true)
            {
                //try to connect to OpenCOVER 
                if (!COVER.Instance.isConnected())
                {
                    string host = Globals.Ribbons.Ribbon1.host.Text;
                    int port = Convert.ToInt32(Globals.Ribbons.Ribbon1.port.Text); 
                    if (COVER.Instance.ConnectToOpenCOVER(host, port))
                    {
                        sendApplicationType();
                        Globals.Ribbons.Ribbon1.connected.Checked = true;
                    }
                    else
                    {
                        Thread.Sleep(3000);
                        Globals.Ribbons.Ribbon1.connected.Checked = false;
                    }
                }
                COVER.Instance.checkForMessages();
                while (COVER.Instance.messageQueue.Count > 0)
                {
                    COVERMessage m = COVER.Instance.messageQueue.Dequeue();
                    if ((COVER.MessageTypes)m.messageType == COVER.MessageTypes.StringMessage)
                    {
                        string input = m.message.readString();
                        if (input.StartsWith("save "))
                        {
                            // FIXME hard coded filename
                            word.save("newdoc");
                            sendStringMessage("OK");
                            System.Console.Error.Flush();
                        }
                        else if (input.StartsWith("gotoPage "))
                        {
                            string[] parameters = input.Split(' ');
                            bool result;
                            if (parameters.Length > 2)
                                result = word.gotoPage(int.Parse(parameters[1]), int.Parse(parameters[2]));
                            else
                                result = word.gotoPage(int.Parse(parameters[1]));

                            if (result)
                                sendStringMessage("OK");
                            else
                                sendStringMessage("FAIL");

                        }
                        else if (input == "createMinutes")
                        {
                            word.createMinutes();
                            sendStringMessage("OK");
                            System.Console.Error.Flush();
                        }
                        else if (input.StartsWith("addPart "))
                        {
                            string[] parts = input.Split(' ');
                            input = input.Remove(0, parts[0].Length + parts[1].Length + 2);

                            if (parts[1].Trim() == "snapshot")
                            {
                                word.addPart(new Word.DocumentPart(parts[1].Trim(), input, Word.DocumentPartType.Picture));
                                sendStringMessage("OK");
                                System.Console.Error.Flush();
                            }
                            else
                            {
                                word.addPart(parts[1].Trim(), input);
                                sendStringMessage("OK");
                                System.Console.Error.Flush();
                            }
                        }
                        else if (input.StartsWith("setPart "))
                        {
                            string[] parts = input.Split(' ');
                            input = input.Remove(0, parts[0].Length + parts[1].Length + 2);
                            word.setPart(parts[1].Trim(), input);
                            sendStringMessage("OK");
                            System.Console.Error.Flush();
                        }
                    }
                    else if ((COVER.MessageTypes)m.messageType == COVER.MessageTypes.MSG_PNGSnapshot)
                    {
                        int numBytes = m.message.readInt();
                        string fn = System.IO.Path.GetTempFileName();
                        fn += (".png");
                        Byte[] bytes = m.message.readBytes(numBytes);
                        ByteArrayToFile(fn, bytes);
                        string transform = m.message.readString();
                        addImage(fn, transform);
                    }
                }
            }
        }
        
    public void addImage(string path,string transform)
    {
        this.Application.Selection.EndKey(Microsoft.Office.Interop.Word.WdUnits.wdStory);

        if (path.Length != 0)
        {
            object tr = Microsoft.Office.Core.MsoTriState.msoTrue;
            //object fa = Microsoft.Office.Core.MsoTriState.msoFalse;
                try
                {
                    Microsoft.Office.Interop.Word.InlineShape picture = this.Application.Selection.InlineShapes.AddPicture(path, ref tr, ref tr, ref optional);

                    picture.AlternativeText = transform;
                }
                catch (Exception e)
                {
                    this.Application.Selection.TypeText(e.Message + ": " + path + "\n");
                }
                this.Application.Selection.TypeText("\nBildunterschrift\n");
        }

    }
        public bool ByteArrayToFile(string _FileName, byte[] _ByteArray)
        {
            try
            {
                // Open file for reading
                System.IO.FileStream _FileStream =
                   new System.IO.FileStream(_FileName, System.IO.FileMode.Create,
                                            System.IO.FileAccess.Write);
                System.IO.BinaryWriter bw = new System.IO.BinaryWriter(_FileStream);
                // Writes a block of bytes to this stream using data from
                // a byte array.
                bw.Write(_ByteArray, 0, _ByteArray.Length);

                // close file stream
                bw.Close();

                return true;
            }
            catch (Exception _Exception)
            {
                // Error
                Console.WriteLine("Exception caught in process: {0}",
                                  _Exception.ToString());
            }

            // error occured, return false
            return false;
        }
        public void sendStringMessage(string str)
        {

            MessageBuffer mb = new MessageBuffer();

            mb.add(str);

            COVER.Instance.sendMessage(mb.buf, COVER.MessageTypes.StringMessage);
        }
        public void sendViewpointMessage(string str)
        {

            MessageBuffer mb = new MessageBuffer();

            mb.add("setViewpoint " + str);

            COVER.Instance.sendMessage(mb.buf, COVER.MessageTypes.StringMessage);
        }
        
        private void ThisAddIn_Startup(object sender, System.EventArgs e)
        {
            coverAddin = this;
            word = new Word();
            word.start();
            DocumentBeforeDoubleClick();
            coverThread = new Thread(new ThreadStart(messageLoop));
            // Start the thread
            coverThread.Start();
        }

        private void ThisAddIn_Shutdown(object sender, System.EventArgs e)
        {
            coverThread.Abort();
        }

        #region VSTO generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InternalStartup()
        {
            this.Startup += new System.EventHandler(ThisAddIn_Startup);
            this.Shutdown += new System.EventHandler(ThisAddIn_Shutdown);
        }
        
        #endregion
    }


    class Word
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
                        ThisAddIn.The.sendStringMessage("Word.ScrollEvent " + page + "/" + line);
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
            this.word = ThisAddIn.The.Application;
            if (this.word == null)
            {
                this.word = new Microsoft.Office.Interop.Word.Application();
            }

            this.word.DocumentOpen += new ApplicationEvents4_DocumentOpenEventHandler(documentOpen);
            this.word.WindowActivate += new ApplicationEvents4_WindowActivateEventHandler(windowActivate);

        }

        void windowActivate(_Document Doc, Window Wn)
        {
            ThisAddIn.The.sendStringMessage("Word.WindowActivateEvent " + Wn.Caption + "/" + Doc.Name);
            this.document = Doc;
        }

        void documentOpen(_Document Doc)
        {
            ThisAddIn.The.sendStringMessage("Word.DocumentOpenEvent " + Doc.Name);
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
                    object msoTrue = Microsoft.Office.Core.MsoTriState.msoTrue;
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
            object goTo = (object)page;

            Range rangeDocument = this.document.Range(ref optional, ref optional);
            Range rangePageBegin = rangeDocument.GoTo(ref goToItem, ref goToDirection, ref goTo, ref optional);

            goToDirection = WdGoToDirection.wdGoToRelative;
            goToItem = WdGoToItem.wdGoToLine;
            goTo = (object)line;

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
