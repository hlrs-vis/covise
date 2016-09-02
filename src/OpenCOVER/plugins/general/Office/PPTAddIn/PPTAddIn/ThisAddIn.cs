using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using System.Threading;
using PowerPoint = Microsoft.Office.Interop.PowerPoint;
using Office = Microsoft.Office.Core;

using OpenCOVERInterface;
using PPTAddIn;
using System.Reflection;

namespace PPTAddIn
{
    public partial class ThisAddIn
    {
        private static ThisAddIn coverAddin = null;
        public static ThisAddIn The
        {
            get { return coverAddin; }
        }
        Thread coverThread;
        PowerPoint ppt;
        private static object optional = Missing.Value;
        public void sendApplicationType()
        {

            MessageBuffer mb = new MessageBuffer();
            mb.add("PowerPoint");
            mb.add("Microsoft PowerPoint Add-in");

            COVER.Instance.sendMessage(mb.buf, COVER.MessageTypes.ApplicationType);
        }
        public void messageLoop()
        {
            while (true)
            {
                //try to connect to OpenCOVER 
                if (!COVER.Instance.isConnected())
                {
                    string host = Globals.Ribbons.Ribbon2.host.Text;
                    int port = Convert.ToInt32(Globals.Ribbons.Ribbon2.port.Text);
                    if (COVER.Instance.ConnectToOpenCOVER(host, port))
                    {
                        sendApplicationType();
                        Globals.Ribbons.Ribbon2.connected.Checked = true;
                    }
                    else
                    {
                        Thread.Sleep(3000);
                        Globals.Ribbons.Ribbon2.connected.Checked = false;
                    }
                }
                COVER.Instance.checkForMessages();
                while (COVER.Instance.messageQueue.Count > 0)
                {
                    COVERMessage m = COVER.Instance.messageQueue.Dequeue();
                    if ((COVER.MessageTypes)m.messageType == COVER.MessageTypes.StringMessage)
                    {
                        string input = m.message.readString();
                        
                        if (input.StartsWith("startSlideShow"))
                        {
                            ppt.startSlideShow();
                            sendStringMessage("OK");
                        }
                        else if (input.StartsWith("stopSlideShow"))
                        {
                            ppt.stopSlideShow();
                            sendStringMessage("OK");
                        }
                        else if (input.StartsWith("next"))
                        {
                            ppt.next();
                            sendStringMessage("OK");
                        }
                        else if (input.StartsWith("previous"))
                        {
                            ppt.previous();
                            sendStringMessage("OK");
                        }
                        else if (input.StartsWith("currentSlide"))
                        {
                            int cs = ppt.CurrentSlide;
                            sendStringMessage("currentSlide " + cs.ToString());
                        }
                        else if (input.StartsWith("quit"))
                        {
                            ppt.quit();
                            sendStringMessage("OK");
                        }
                        else if (input.StartsWith("load"))
                        {
                            string[] parameters = input.Split(' ');
                            ppt.load(parameters[1]);
                            sendStringMessage("OK");
                        }
                        else if (input.StartsWith("save"))
                        {
                            string[] parameters = input.Split(' ');
                            ppt.save(parameters[1]);
                            sendStringMessage("OK");
                        }
                        else if (input.StartsWith("setCurrentSlide"))
                        {
                            string[] parameters = input.Split(' ');
                            if (parameters.Length > 1)
                            {
                                int cs = ppt.CurrentSlide = int.Parse(parameters[1]);
                                sendStringMessage(cs.ToString());
                            }
                        }
                        
                        /*else if (input.StartsWith("gotoPage "))
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

                        }*/
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

        public void addImage(string path, string transform)
        {
            Microsoft.Office.Interop.PowerPoint.Presentation p = this.Application.ActivePresentation;
            int numberOfSlides =p.Slides.Count;
            numberOfSlides++;

            p.Slides.Add(numberOfSlides,Microsoft.Office.Interop.PowerPoint.PpSlideLayout.ppLayoutObjectAndText);
            int numShapes = p.Slides[numberOfSlides].Shapes.Count;
            p.Slides[numberOfSlides].Shapes.AddPicture(path,Microsoft.Office.Core.MsoTriState.msoTrue,Microsoft.Office.Core.MsoTriState.msoTrue,0,0);
            Microsoft.Office.Interop.PowerPoint.Shape s = p.Slides[numberOfSlides].Shapes[2]; // Shape 2 is the image, 1 the title and 3 the items list to the right
            s.AlternativeText = transform; 
            try
            {
                var codeModule = p.VBProject.VBComponents.Add(Microsoft.Vbe.Interop.vbext_ComponentType.vbext_ct_StdModule);
                StringBuilder moduleCode = new StringBuilder();
                moduleCode.AppendLine("Sub ShowInfo(s AS String)");
                moduleCode.AppendLine("\t" + @"Msgbox s");
                moduleCode.AppendLine("End Sub");
                codeModule.CodeModule.AddFromString(moduleCode.ToString());
                s.ActionSettings[Microsoft.Office.Interop.PowerPoint.PpMouseActivation.ppMouseClick].Action = Microsoft.Office.Interop.PowerPoint.PpActionType.ppActionRunMacro;
                s.ActionSettings[Microsoft.Office.Interop.PowerPoint.PpMouseActivation.ppMouseClick].Run = "ShowInfo(\"" + transform + "\")";
            }
            catch
            {
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
            ppt = new PowerPoint();
            ppt.start();
            //DocumentBeforeDoubleClick();
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
}
