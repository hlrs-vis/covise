using System;
using System.Collections.Generic;
using System.Text;


namespace OpenCOVERPlugin
{
   public class MessageBuffer
   {
      public byte[] buf;
      int currentPos;
      System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
      public MessageBuffer()
      {
         currentPos = 0;
         buf = new byte[256];
      }
      public MessageBuffer(byte[] b)
      {
          currentPos = 0;
          buf = b;
      }
      private void incsize(int s)
      {
         if (buf.Length < currentPos + s)
         {
            byte[] newbuf = new byte[buf.Length + s + 1024];
            buf.CopyTo(newbuf, 0);
            buf = newbuf;
         }
      }
      public bool readBool()
      {
          bool b = (buf[currentPos] == 1);
          currentPos += 1;
          return b;
      }
      public byte readByte()
      {
          byte b = buf[currentPos];
          currentPos += 1;
          return b;
      }

      public Autodesk.Revit.DB.Color readColor()
      {
          byte r, g, b;
          r = buf[currentPos];
          currentPos += 1;
          g = buf[currentPos];
          currentPos += 1;
          b = buf[currentPos];
          currentPos += 1;
          return new Autodesk.Revit.DB.Color(r,g,b);
      }
      public Autodesk.Revit.DB.XYZ readXYZ()
      {
          double x = readDouble();
          double y = readDouble();
          double z = readDouble();
          Autodesk.Revit.DB.XYZ c = new Autodesk.Revit.DB.XYZ(x, y, z);
          return c;
      }
      public int readInt()
      {
          int i = BitConverter.ToInt32(buf, currentPos);
          currentPos += 4;
          return i;
      }
      public float readFloat()
      {
          float i = BitConverter.ToSingle(buf, currentPos);
          currentPos += 4;
          return i;
      }
      public double readDouble()
      {
          double i = BitConverter.ToDouble(buf, currentPos);
          currentPos += 8;
          return i;
      }
      public string readString()
      {
          string str;
          int len=0;
          while (buf[currentPos+len] != '\0')
          {
              len++;
          }
          str = enc.GetString(buf, currentPos,len);
          currentPos += len+1;
          return str;
      }

      public void add(bool num)
      {
         incsize(1);
         if (num)
            buf[currentPos] = 1;
         else
            buf[currentPos] = 0;
         currentPos += 1;
      }
      public void add(byte b)
      {
         incsize(1);
         buf[currentPos] = b;
         currentPos += 1;
      }
      public void add(Autodesk.Revit.DB.Color c)
      {
         if (c == null)
         {
             add(0);
             add(0);
             add(0);
         }
         else
         {
             add(c.Red);
             add(c.Green);
             add(c.Blue); 
         }
      }
      public void add(Autodesk.Revit.DB.XYZ c)
      {
          if (c == null)
          {
              add((float)0);
              add((float)0);
              add((float)0);
          }
          else
          {
              add((float)c.X);
              add((float)c.Y);
              add((float)c.Z);
          }
      }
      public void add(int num)
      {
         incsize(4);

         BitConverter.GetBytes(num).CopyTo(buf, currentPos);
         currentPos += 4;
      }
      public void add(float num)
      {
         incsize(4);
         BitConverter.GetBytes(num).CopyTo(buf, currentPos);
         currentPos += 4;
      }
      public void add(double num)
      {
         incsize(8);
         BitConverter.GetBytes(num).CopyTo(buf, currentPos);
         currentPos += 8;
      }
      public void add(string str)
      {
          if (str != null)
          {
              byte[] ba = enc.GetBytes(str);
              incsize(ba.Length + 1);
              for (int i = 0; i < ba.Length; i++)
              {
                  buf.SetValue(ba[i], currentPos);
                  currentPos++;
              }
              buf[currentPos] = 0;
              currentPos++;
          }
          else
          {
              incsize(1);
              buf[currentPos] = 0;
              currentPos++;
          }
      }
   }
}
