using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Linq;
using IWin32Window = System.Windows.Forms.IWin32Window;


namespace OpenCOVERInterface
{

    public class COVERMessage
    {

        public MessageBuffer message;
        public int messageType;
        public COVERMessage(MessageBuffer m, int mt)
        {
            message = m;
            messageType = mt;
        }
    }

   public sealed class COVER
   {

       public enum MessageTypes { StringMessage = 700, ApplicationType };
      private Thread messageThread;

      private System.Net.Sockets.TcpClient toCOVER;
      public Queue<COVERMessage> messageQueue;


      COVER()
      {
      }
      /// <summary>
      /// Singleton class which holds the connection to OpenCOVER, is used to communicate with OpenCOVER
      /// </summary>
      public static COVER Instance
      {
         get
         {
            return Nested.instance;
         }
      }

      public bool isConnected()
      {
          return (toCOVER != null);
      }
      public void handleMessages()
      {
          if (toCOVER != null)
          {
              Byte[] data = new Byte[2000];
              while (true)
              {
                  int len = 0;
                  while (len < 16)
                  {
                      int numRead;
                      try
                      {
                          numRead = toCOVER.GetStream().Read(data, len, 16 - len);
                      }
                      catch (System.IO.IOException e)
                      {
                          // probably socket closed
                          return;
                      }
                      len += numRead;
                  }

                  int msgType = BitConverter.ToInt32(data, 2 * 4);
                  int length = BitConverter.ToInt32(data, 3 * 4);
                  length = (int)ByteSwap.swap((uint)length);
                  msgType = (int)ByteSwap.swap((uint)msgType);
                  len = 0;
                  while (len < length)
                  {
                      int numRead;
                      try
                      {
                          numRead = toCOVER.GetStream().Read(data, len, length - len);
                      }
                      catch (System.IO.IOException e)
                      {
                          // probably socket closed
                          return;
                      }
                      len += numRead;
                  }
                  COVERMessage m = new COVERMessage(new MessageBuffer(data), msgType);
                  messageQueue.Enqueue(m);
              }
          }
      }
      public bool checkForMessages()
      {
          int len = 0;
          Byte[] data = new Byte[16];

          while (len < 16)
          {
              int numRead;
              try
              {
                  if (toCOVER != null)
                  {
                      //toCOVER.GetStream().ReadTimeout = 10;
                      numRead = toCOVER.GetStream().Read(data, len, 16 - len);
                      if (numRead == 0 && len == 0)
                          return false;
                  }
                  else
                  {
                      return false;
                  }
              }
              catch (System.IO.IOException e)
              {
                  toCOVER = null;
                  // probably socket closed
                  return false;
              }
              catch (System.InvalidOperationException e)
              {
                  toCOVER = null;
                  // probably socket closed
                  return false;
              }
              len += numRead;
          }

          int msgType = BitConverter.ToInt32(data, 2 * 4);
          int length = BitConverter.ToInt32(data, 3 * 4);
          length = (int)ByteSwap.swap((uint)length);
          msgType = (int)ByteSwap.swap((uint)msgType);

          data = new Byte[length];
          len = 0;
          while (len < length)
          {
              int numRead;
              try
              {
                  numRead = toCOVER.GetStream().Read(data, len, length - len);
              }
              catch (System.IO.IOException e)
              {
                  // probably socket closed
                  return false;
              }
              len += numRead;
          }
          COVERMessage m = new COVERMessage(new MessageBuffer(data), msgType);
          messageQueue.Enqueue(m);
          return true;
      }

      public bool ConnectToOpenCOVER(string host, int port)
      {
         messageQueue = new Queue<COVERMessage>();

         try
         {
             if (toCOVER != null)
             {
                 messageThread.Abort(); // stop reading from the old toCOVER connection
                 toCOVER.Close();
                 toCOVER = null;
             }

             toCOVER = new TcpClient(host, port);
             if (toCOVER.Connected)
             {
                 // Sends data immediately upon calling NetworkStream.Write.
                 toCOVER.NoDelay = true;
                 LingerOption lingerOption = new LingerOption(false, 0);
                 toCOVER.LingerState = lingerOption;

                 NetworkStream s = toCOVER.GetStream();
                 Byte[] data = new Byte[256];
                 data[0] = 1;
                 try
                 {
                     //toCOVER.ReceiveTimeout = 1000;
                     s.Write(data, 0, 1);
                     //toCOVER.ReceiveTimeout = 10000;
                 }
                 catch (System.IO.IOException e)
                 {
                     // probably socket closed
                     toCOVER = null;
                     return false;
                 }

                 int numRead = 0;
                 try
                 {
                     //toCOVER.ReceiveTimeout = 1000;
                     numRead = s.Read(data, 0, 1);
                     //toCOVER.ReceiveTimeout = 10000;
                 }
                 catch (System.IO.IOException e)
                 {
                     // probably socket closed
                     toCOVER = null;
                     return false;
                 }
                 if (numRead == 1)
                 {

                    // messageThread = new Thread(new ThreadStart(this.handleMessages));

                     // Start the thread
                     //messageThread.Start();

                 }

                 return true;
             }
             //String errorMessage = "Could not connect to OpenCOVER on "+ host+ ", port "+ Convert.ToString(port);
             //System.Windows.Forms.MessageBox.Show(errorMessage);
         }
         catch
         {
             //String errorMessage = "Connection error while trying to connect to OpenCOVER on " + host + ", port " + Convert.ToString(port);
             //System.Windows.Forms.MessageBox.Show(errorMessage);

         }
         toCOVER = null;
         return false;

      }

      public void sendMessage(Byte[] messageData, OpenCOVERInterface.COVER.MessageTypes msgType)
      {
         int len = messageData.Length + (4 * 4);
         Byte[] data = new Byte[len];
         ByteSwap.swapTo((uint)msgType, data, 2 * 4);
         ByteSwap.swapTo((uint)messageData.Length, data, 3 * 4);
         messageData.CopyTo(data, 4 * 4);
         toCOVER.GetStream().Write(data, 0, len);
      }

      class Nested
      {
         // Explicit static constructor to tell C# compiler
         // not to mark type as beforefieldinit
         static Nested()
         {
         }
         internal static readonly COVER instance = new COVER();
      }
   }

}
