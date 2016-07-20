using System;
using System.Collections.Generic;
using System.Text;

namespace OpenCOVERInterface
{
   class ByteSwap
   {
      public static UInt16 swap(UInt16 input)
      {
         return ((UInt16)(
         ((0xFF00 & input) >> 8) |
         ((0x00FF & input) << 8)));
      }

      public static UInt32 swap(UInt32 input)
      {
         return ((UInt32)(
         ((0xFF000000 & input) >> 24) |
         ((0x00FF0000 & input) >> 8) |
         ((0x0000FF00 & input) << 8) |
         ((0x000000FF & input) << 24)));
      }

      public static void swapTo(int input, Array dest, int pos)
      {
         swapTo((uint)input, dest, pos);
      }
      public static void swapTo(UInt32 input, Array dest, int pos)
      {
         UInt32 swapped = (
         ((0xFF000000 & input) >> 24) |
         ((0x00FF0000 & input) >> 8) |
         ((0x0000FF00 & input) << 8) |
         ((0x000000FF & input) << 24));
         BitConverter.GetBytes(swapped).CopyTo(dest, pos);
      }
      public static void swapTo(float input, Array dest, int pos)
      {
         byte[] tmpIn = BitConverter.GetBytes(input);
         dest.SetValue(tmpIn[3], pos);
         dest.SetValue(tmpIn[2], pos + 1);
         dest.SetValue(tmpIn[1], pos + 2);
         dest.SetValue(tmpIn[0], pos + 3);
      }
      public static float swap(float input)
      {
         byte[] tmpIn = BitConverter.GetBytes(input);
         byte[] tmpOut = new byte[4];
         tmpOut[0] = tmpIn[3];
         tmpOut[1] = tmpIn[2];
         tmpOut[2] = tmpIn[1];
         tmpOut[3] = tmpIn[0];
         return BitConverter.ToSingle(tmpOut, 0);
      }

      public static double swap(double input)
      {
         byte[] tmpIn = BitConverter.GetBytes(input);
         byte[] tmpOut = new byte[8];
         tmpOut[0] = tmpIn[7];
         tmpOut[1] = tmpIn[6];
         tmpOut[2] = tmpIn[5];
         tmpOut[3] = tmpIn[4];
         tmpOut[4] = tmpIn[3];
         tmpOut[5] = tmpIn[2];
         tmpOut[6] = tmpIn[1];
         tmpOut[7] = tmpIn[0];
         return BitConverter.ToDouble(tmpOut, 0);
      }
   }
}
