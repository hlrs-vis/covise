using System;
using System.Collections.Generic;
using System.Text;


namespace OpenCOVERPlugin
{
    public class MessageBuffer
    {
        public enum Types
        {
            TbBool = 7,
            TbInt64,
            TbInt32,
            TbFloat,
            TbDouble,
            TbString,
            TbChar,
            TbTB,
            TbBinary,
        };

        public byte[] buf;
        int currentPos;
        bool typeInfo = false;
        System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
        public MessageBuffer()
        {
            currentPos = 1;
            buf = new byte[256];
            if (typeInfo)
                buf[0] = 1; // no type info in TokenBuffer
            else
                buf[0] = 0; // no type info in TokenBuffer
        }
        public MessageBuffer(byte[] b)
        {
            currentPos = 1;
            buf = new byte[b.Length];
            if (buf[0] == 0)
                typeInfo = false;
            else
                typeInfo = true;
            b.CopyTo(buf, 0);
        }
        private void readTypeInfo(Types expectedType)
        {
            if(typeInfo)
            {
                if(buf[currentPos] != (byte)expectedType)
                {
                    // Todo print error message
                    System.Threading.Thread.Sleep(5000);
                }
                currentPos++;
            }
        }
        private void addTypeInfo(Types followingType)
        {
            if (typeInfo)
            {
                incsize(1);
                buf[currentPos] = (byte)followingType;
                currentPos++;
            }
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
            readTypeInfo(Types.TbBool);
            bool b = (buf[currentPos] == 1);
            currentPos += 1;
            return b;
        }
        public byte readByte()
        {
            readTypeInfo(Types.TbChar);
            byte b = buf[currentPos];
            currentPos += 1;
            return b;
        }

        public Autodesk.Revit.DB.Color readColor()
        {
            byte r, g, b;
            readTypeInfo(Types.TbChar);
            r = buf[currentPos];
            currentPos += 1;
            readTypeInfo(Types.TbChar);
            g = buf[currentPos];
            currentPos += 1;
            readTypeInfo(Types.TbChar);
            b = buf[currentPos];
            currentPos += 1;
            return new Autodesk.Revit.DB.Color(r, g, b);
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
            readTypeInfo(Types.TbInt32);
            int i = BitConverter.ToInt32(buf, currentPos);
            currentPos += 4;
            return i;
        }
        public float readFloat()
        {
            readTypeInfo(Types.TbFloat);
            float i = BitConverter.ToSingle(buf, currentPos);
            currentPos += 4;
            return i;
        }
        public double readDouble()
        {
            readTypeInfo(Types.TbDouble);
            double i = BitConverter.ToDouble(buf, currentPos);
            currentPos += 8;
            return i;
        }
        public string readString()
        {
            readTypeInfo(Types.TbString);
            string str;
            int len = 0;
            while (buf[currentPos + len] != '\0')
            {
                len++;
            }
            str = enc.GetString(buf, currentPos, len);
            currentPos += len + 1;
            return str;
        }

        public void add(bool num)
        {
            addTypeInfo(Types.TbChar);
            incsize(1);
            if (num)
                buf[currentPos] = 1;
            else
                buf[currentPos] = 0;
            currentPos += 1;
        }
        public void add(byte b)
        {
            addTypeInfo(Types.TbChar);
            incsize(1);
            buf[currentPos] = b;
            currentPos += 1;
        }
        public void add(TextureInfo ti)
        {
            add(ti.textuerPath);
            add(ti.sx);
            add(ti.sy);
            add(ti.ox);
            add(ti.oy);
            add(ti.angle);
            add(ti.color);
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
            addTypeInfo(Types.TbInt32);
            incsize(4);

            BitConverter.GetBytes(num).CopyTo(buf, currentPos);
            currentPos += 4;
        }
        public void add(float num)
        {
            addTypeInfo(Types.TbFloat);
            incsize(4);
            BitConverter.GetBytes(num).CopyTo(buf, currentPos);
            currentPos += 4;
        }
        public void add(double num)
        {
            addTypeInfo(Types.TbDouble);
            incsize(8);
            BitConverter.GetBytes(num).CopyTo(buf, currentPos);
            currentPos += 8;
        }
        public void add(string str)
        {
            addTypeInfo(Types.TbDouble);
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

        public void add(byte[] ba)
        {
            addTypeInfo(Types.TbBinary);
            incsize(ba.Length);
            for (int i = 0; i < ba.Length; i++)
            {
                buf.SetValue(ba[i], currentPos);
                currentPos++;
            }
        }
    }
}
