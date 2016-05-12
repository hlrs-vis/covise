using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Reflection;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using OpenCOVERInterface;

namespace OfficeConsole
{
  class OfficeConsole
  {
    public Object lockvar = ""; // Lock-Variable

    private Thread readThread = null;

    private Queue<string> inputQueue = new Queue<string>();
    private Queue<string> outputQueue = new Queue<string>();

    private Excel excel = null;
    private InternetExplorer ie = null;
    private PowerPoint powerpoint = null;
    private Word word = null;

    private BasicOfficeControl control = null;

    private bool run = true;

    private static OfficeConsole officeConsole = null;
    public static OfficeConsole The
    {
      get { return officeConsole; }
    }
    private int port;
    private string host;

    public enum Mode
    {
      Unknown = 0x00,
      Excel,
      InternetExplorer,
      PowerPoint,
      Word
    };

    Mode mode = Mode.Unknown;

    public void setCOVERConnectionInfo(string h, int p)
    {
        host = h;
        port = p;
    }

    public OfficeConsole(OfficeConsole.Mode mode)
    {
      this.mode = mode;
      OfficeConsole.officeConsole = this;

      switch (mode)
      {
        case Mode.InternetExplorer:
          this.ie = new InternetExplorer();
          this.control = this.ie;
          this.ie.start();
          break;
        case Mode.Word:
          this.word = new Word();
          this.control = this.word;
          this.word.start();
          break;
        case Mode.PowerPoint:
          this.powerpoint = new PowerPoint();
          this.control = this.powerpoint;
          this.powerpoint.start();
          break;
        case Mode.Excel:
          this.excel = new Excel();
          this.control = this.excel;
          this.excel.start();
          break;
        default:
          Application.Exit();
          break;
      }

      //readThread = new Thread(new ParameterizedThreadStart(this.reader));
      //readThread.Start();

    }
    public void sendApplicationType()
    {

        MessageBuffer mb = new MessageBuffer();
        if (mode == Mode.Word)
        {
            mb.add("Word");
            mb.add("Microsoft Word");
        }
        if (mode == Mode.Excel)
        {
            mb.add("Calc");
            mb.add("Excel");
        }
        if (mode == Mode.PowerPoint)
        {
            mb.add("Presentation");
            mb.add("PowerPoint");
        }
        if (mode == Mode.InternetExplorer)
        {
            mb.add("Browser");
            mb.add("InternetExplorer");
        }
        
        COVER.Instance.sendMessage(mb.buf, COVER.MessageTypes.ApplicationType);
    }
    public void sendStringMessage(string str)
    {

        MessageBuffer mb = new MessageBuffer();

        mb.add(str);

        COVER.Instance.sendMessage(mb.buf, COVER.MessageTypes.StringMessage);
    }
    public void start()
    {

      string input = "";

      while (run)
      {

          //try to connect to OpenCOVER 
          if (!COVER.Instance.isConnected())
          {
              if (COVER.Instance.ConnectToOpenCOVER(host, port))
              {
                  sendApplicationType();
              }
              else
              {
                  Thread.Sleep(3000);
              }
          }
        Monitor.Enter(inputQueue);
        int count = inputQueue.Count;
        if (count > 0)
          input = inputQueue.Dequeue().Trim();
        Monitor.Exit(inputQueue);

        if (count > 0)
        {

          // Common commands
          if (input == "start")
          {
            control.start();
            sendStringMessage("OK");
          }

          else if (input == "quit")
          {
            control.quit();
            run = false;
            sendStringMessage("OK");
          }

          else if (input.StartsWith("load "))
          {
            string file = input.Remove(0, 4).Trim();
            control.load(file);
            sendStringMessage("OK");
          }


          // Internet Explorer
          if (this.mode == Mode.InternetExplorer)
          {
            if (input == "forward")
            {
              ie.forward();
              sendStringMessage("OK");
            }
            else if (input == "back")
            {
              ie.back();
              sendStringMessage("OK");
            }
            else if (input.Trim().StartsWith("navigate2"))
            {
              string url = input.Remove(0, 9);
              ie.navigate2(url.Trim());
              sendStringMessage("OK");
            }
          }

          // Word
          else if (this.mode == Mode.Word)
          {
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

          // PowerPoint
          else if (this.mode == Mode.PowerPoint)
          {
            if (input == "startSlideShow")
            {
              if (powerpoint.startSlideShow())
                sendStringMessage("OK");
              else
                sendStringMessage("FAIL");
            }
            else if (input == "stopSlideShow")
            {
              if (powerpoint.stopSlideShow())
                sendStringMessage("OK");
              else
                sendStringMessage("FAIL");

            }
            else if (input.StartsWith("setCurrentSlide"))
            {
              string number = input.Remove(0, 15);
              int slideNumber;
              try
              {
                slideNumber = Convert.ToInt32(number);
                powerpoint.CurrentSlide = slideNumber;
                if (powerpoint.CurrentSlide == slideNumber)
                  sendStringMessage("OK");
                else
                  sendStringMessage("FAIL");
              }
              catch
              {
                sendStringMessage("FAIL");
              }
            }
            else if (input == "getCurrentSlide")
            {
               sendStringMessage(Convert.ToString(powerpoint.CurrentSlide));
            }
            else if (input == "next")
            {
                powerpoint.next();
            }
            else if (input == "previous")
            {
                powerpoint.previous();
            }

          }

          else if (this.mode == Mode.Excel)
          {
            if (input.StartsWith("createWorksheet"))
            {
              string str = input.Remove(0, 16);
              excel.createWorksheet(str.Trim());
              sendStringMessage("OK");
            }
            else if (input.StartsWith("activateWorksheet "))
            {
              string str = input.Remove(0, 18);
              if (excel.activateWorksheet(str.Trim()))
                sendStringMessage("OK");
              else
                sendStringMessage("FAIL");
            }
            else if (input.StartsWith("activateWorkbook "))
            {
              string str = input.Remove(0, 17);
              if (excel.activateWorkbook(str.Trim()))
                sendStringMessage("OK");
              else
                sendStringMessage("FAIL");
            }
            else if (input.StartsWith("runMacro "))
            {
              string str = input.Remove(0, 9);
              if (excel.runMacro(str.Trim()))
                sendStringMessage("OK");
              else
                sendStringMessage("FAIL");
            }
            else if (input.StartsWith("getRange "))
            {
              string str = input.Remove(0, 9).Trim();
              str = str.Replace('!', ':');
              string[] fromTo = str.Split(':');
              string[,] result = null;

              if (fromTo.Length < 2)
                sendStringMessage("");
              else
              {
                if (fromTo.Length == 2)
                {
                  result = this.excel.getRange(fromTo[0], fromTo[1]);
                }
                else
                {
                  result = this.excel.getRange(fromTo[0], fromTo[1], fromTo[2]);
                }
              }

              if (result == null)
              {
                sendStringMessage("");
              }
              else
              {
                string output = "";

                foreach (string entry in result)
                {
                  output += entry + "|";
                }
                char[] trim = { '|' };
                output = output.TrimEnd(trim);
                sendStringMessage(output);
              }
            }


          }
          else if (input == "exit")
          {
            run = false;
            sendStringMessage("OK");
          }
        }
        else
        {
          this.control.update();
          Monitor.Enter(this.outputQueue);
          if (outputQueue.Count > 0)
            System.Console.Out.WriteLine(outputQueue.Dequeue().Trim());
          Monitor.Exit(this.outputQueue);
          COVER.Instance.checkForMessages();
          while (COVER.Instance.messageQueue.Count > 0)
          {
              COVERMessage m = COVER.Instance.messageQueue.Dequeue();
              if ((COVER.MessageTypes)m.messageType == COVER.MessageTypes.StringMessage)
              {
                  input = m.message.readString();
                  inputQueue.Enqueue(input);
              }
              else if ((COVER.MessageTypes)m.messageType == COVER.MessageTypes.MSG_PNGSnapshot)
              {
                  int numBytes = m.message.readInt();
                  string fn = System.IO.Path.GetTempFileName();
                  fn +=(".png");
                  Byte[] bytes = m.message.readBytes(numBytes);
                  ByteArrayToFile(fn,bytes);
                  if (mode == Mode.Word)
                  {
                      word.addImage(fn);
                  }
              }
          }
          //Thread.Sleep(10);
        }
      }
      //readThread.Abort();
      control.quit();

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
    public void sendEvent(string eventDescription)
    {
      sendStringMessage(eventDescription);
    }

    // THREAD /////////////////////////////////////////////////////////////////////////////////////////

    public void reader(object o)
    {
      bool run = true;
      while (run)
      {
        string input = System.Console.In.ReadLine();
        Monitor.Enter(inputQueue);
        inputQueue.Enqueue(input);
        Monitor.Exit(inputQueue);

        if (input == "exit" || input == "quit")
        {
          run = false;
        }
      }

    }

    // MAIN ///////////////////////////////////////////////////////////////////////////////////////////

    static void Main(string[] args)
    {
      string cmd = "";
      if (args.Length < 1)
      {
          Console.WriteLine("Usage: MSOfficeInterface [host] [-p port]-IE | -Word | -PowerPoint | -Excel");
          return;
      }
      OfficeConsole console = null;
      string host="localhost";
      int port=31315;
      for (int argnum = 0; argnum < args.Length; argnum++)
      {

          cmd = args[argnum];

          //System.Diagnostics.Debugger.Break();

          switch (cmd)
          {
              case "-IE":
                  console = new OfficeConsole(Mode.InternetExplorer);
                  break;
              case "-Word":
                  console = new OfficeConsole(Mode.Word);
                  break;
              case "-PowerPoint":
                  console = new OfficeConsole(Mode.PowerPoint);
                  break;
              case "-Excel":
                  console = new OfficeConsole(Mode.Excel);
                  break;
              case "-p":
                  if (argnum < args.Length)
                  {
                      argnum++;
                      port = Convert.ToInt32(args[argnum]);
                  }
                  break;
              default:
                  host = cmd;
                  break;
          }
      }

      if (console == null)
      {
          Console.WriteLine("Usage: MSOfficeInterface [host] [-p port]-IE | -Word | -PowerPoint | -Excel");
          return;
      }
      console.setCOVERConnectionInfo(host, port);
      console.start();

    }

  }
}
