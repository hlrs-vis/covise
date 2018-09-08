// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <iostream>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vvplatform.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvdebugmsg.h"
#include "vvdicom.h"

using namespace std;

//----------------------------------------------------------------------------
/// Constructor
vvDicom::vvDicom(vvDicomProperties* prop)
{
  info = prop;
  explicitVR              = 0;
  photometric_specified   = 0;
  red_table_size          = 0;
  green_table_size        = 0;
  blue_table_size         = 0;
  window_center_specified = 0;
  window_width_specified  = 0;
  slope_specified         = 0;
  intercept_specified     = 0;
  fp = NULL;
}

//----------------------------------------------------------------------------
/** This function determines byte ordering i.e. low-high or high-low.
 recall that little endian == least signicant to most significant byte.
 @return true if little endian
*/
void vvDicom::setEndian()
{
  int    l = 1;
  uchar* u = (uchar*)&l;
  info->littleEndian = (u[0]==1) ? true : false;
  if (vvDebugMsg::isActive(2))
  {
    cerr << "CPU type: ";
    if (info->littleEndian) cerr << "little endian" << endl;
    else cerr << "big endian" << endl;
  }
}

//----------------------------------------------------------------------------
/// This function reads a 16-bit entity/value.
inline int vvDicom::read16(FILE* _fp)
{
  uchar t1, t2;
  size_t n;
  (void)n;
  n = fread(&t1, sizeof t1, 1, _fp);
  assert(n == 1);
  n = fread(&t2, sizeof t2, 1, _fp);
  assert(n == 1);

  if (info->littleEndian)  return t1 + t2*256;
  else                     return t1*256 + t2;
}

//----------------------------------------------------------------------------
/// This function reads a 32-bit entity/value.
inline int vvDicom::read32(FILE* _fp)
{
  uchar t1, t2, t3, t4;
  size_t n;
  (void)n;
  n = fread(&t1, sizeof t1, 1, _fp); assert(n == 1);
  n = fread(&t2, sizeof t2, 1, _fp); assert(n == 1);
  n = fread(&t3, sizeof t3, 1, _fp); assert(n == 1);
  n = fread(&t4, sizeof t4, 1, _fp); assert(n == 1);

  if (info->littleEndian) return t1 + t2*256 + t3*256*256 + t4*256*256*256;
  else                    return t1*256*256*256 + t2*256*256 + t3*256 + t4;
}

//----------------------------------------------------------------------------
/** This functions sifts through the input dicom data.  It's the heart of
  the program and needs to be modified when new fields need to be supported.
*/
void vvDicom::readDicomData(FILE* _fp)
{
  int first_one = 1;
  bool done = false;
  int group, element, e_len;
  int tmp;
  const char* infoText;
  long where;                                     // remember offset to group
  long pos;                                       // temporary position in file

  while(!done)
  {
    infoText = NULL;
    DicomTypes t = VV_UNKNOWN;

    where = ftell(_fp);                            // remember offset to group
    group   = read16(_fp);
    element = read16(_fp);
    e_len   = read32(_fp);

    // check and see if assumed byte order is correct & fix if necessary

	// check for Sequence and DataItem delimiters
	if(group == 0xfffe) // delimiters
	{
		if(element == 0xe0dd) // Sequence Delimition Tag 
		{
			continue;
		}
		if(element == 0xe00d) // Item  Delimition Tag
		{
			continue;
		}
		if(element == 0xe000) // Item 
		{
			continue;
		}
		if(element == 0xfffa) // Digital Signatures Sequence 
		{
			continue;
		}
	}
    if (first_one)
    {
      if (group != 0 && element == 0 && e_len == 4)
      {
        first_one = 0;
      }
      else if (e_len == 0x04000000)
      {
        first_one = 0;
        vvDebugMsg::msg(2, "\nPossible byte ordering problem - switching!");
        info->littleEndian = !info->littleEndian;
        fseek(_fp, 0, 0);
        group   = read16(_fp);
        element = read16(_fp);
        e_len   = read32(_fp);
      }
      else
      {
        first_one = 0;

        // Try DICOM part 10 i.e. a 128 byte file preamble followed by DICM:

        fseek(_fp, 128, 0);                        //skip the preamble - next 4 bytes should
        //be "DICM"
        uchar tt[4];
        size_t n = fread(tt, sizeof tt, 1, _fp);
        if(n != 1)
          cerr << "Returned value from fread is " << n << ", expected 1" << endl;
        assert(n == 1);
        if (tt[0] != 'D' || tt[1] != 'I' || tt[2] != 'C' || tt[3] != 'M')
        {
          // It's not proper part 10 - try w/out the 128 byte preamble:
          fseek(_fp, 0, 0);
          n = fread(tt, sizeof(tt), 1, _fp);
          assert(n == 1);
        }

        if (tt[0] != 'D' || tt[1] != 'I' || tt[2] != 'C' || tt[3] != 'M')
        {
          cerr << "Not a proper DICOM part 10 file." << endl;
          if ( (group == 0 && element == 0 && e_len == 4) ||
            (group == 8 && element == 1)               ||
            (group == 8 && element == 5)               ||
            (group == 8 && element == 8)               ||
            (group == 8 && element == 0x16) )
          {
            first_one = 0;
            fseek(_fp, 0, 0);
            group   = read16(_fp);
            element = read16(_fp);
            e_len   = read32(_fp);
          }
          else assert(0);
        }
        else
        {
          where = ftell(_fp);                      // remember offset to group
          group   = read16(_fp);
          element = read16(_fp);
          e_len   = read32(_fp);

		  if(group == 0xfffc && element == 0xfffc) // Padding
		  {
		  }

          // Check the value representation:
          char* pvr = (char*)&e_len;
          if      ( (pvr[0]=='O' && pvr[1]=='B') ||
            (pvr[0]=='O' && pvr[1]=='W') ||
            (pvr[0]=='S' && pvr[1]=='Q') ||
            (pvr[3]=='O' && pvr[2]=='B') ||
            (pvr[3]=='O' && pvr[2]=='W') ||
            (pvr[3]=='S' && pvr[2]=='Q') )
          {
            // These explicit vr's have a 32-bit length. Therefore, we'll need to read another 16 bits
            // and then include these additional 16 bits in the current 16 bit length.
            explicitVR = 1;
            e_len = read32(_fp);
          }
          else if ( (pvr[0]=='A' && pvr[1]=='E') ||
            (pvr[0]=='A' && pvr[1]=='S') ||
            (pvr[0]=='A' && pvr[1]=='T') ||
            (pvr[0]=='C' && pvr[1]=='S') ||
            (pvr[0]=='D' && pvr[1]=='A') ||
            (pvr[0]=='D' && pvr[1]=='S') ||
            (pvr[0]=='D' && pvr[1]=='T') ||
            (pvr[0]=='F' && pvr[1]=='L') ||
            (pvr[0]=='F' && pvr[1]=='D') ||
            (pvr[0]=='I' && pvr[1]=='S') ||
            (pvr[0]=='L' && pvr[1]=='O') ||
            (pvr[0]=='L' && pvr[1]=='T') ||
            (pvr[0]=='P' && pvr[1]=='N') ||
            (pvr[0]=='S' && pvr[1]=='H') ||
            (pvr[0]=='S' && pvr[1]=='L') ||
            (pvr[0]=='S' && pvr[1]=='S') ||
            (pvr[0]=='S' && pvr[1]=='T') ||
            (pvr[0]=='T' && pvr[1]=='M') ||
            (pvr[0]=='U' && pvr[1]=='I') ||
            (pvr[0]=='U' && pvr[1]=='L') ||
            (pvr[0]=='U' && pvr[1]=='S') )
          {
            // These explicit VR's have a 16-bit length. This
            // allows them to fit into the same space as implicit VR.
            explicitVR = 1;
            if (info->littleEndian)
            {
              e_len &= 0xffff0000;
              e_len >>= 16;
            }
            else
            {
              e_len &= 0x0000ffff;
              e_len <<= 16;
            }
          }
          else if ( (pvr[3]=='A' && pvr[2]=='E') ||
            (pvr[3]=='A' && pvr[2]=='S') ||
            (pvr[3]=='A' && pvr[2]=='T') ||
            (pvr[3]=='C' && pvr[2]=='S') ||
            (pvr[3]=='D' && pvr[2]=='A') ||
            (pvr[3]=='D' && pvr[2]=='S') ||
            (pvr[3]=='D' && pvr[2]=='T') ||
            (pvr[3]=='F' && pvr[2]=='L') ||
            (pvr[3]=='F' && pvr[2]=='D') ||
            (pvr[3]=='I' && pvr[2]=='S') ||
            (pvr[3]=='L' && pvr[2]=='O') ||
            (pvr[3]=='L' && pvr[2]=='T') ||
            (pvr[3]=='P' && pvr[2]=='N') ||
            (pvr[3]=='S' && pvr[2]=='H') ||
            (pvr[3]=='S' && pvr[2]=='L') ||
            (pvr[3]=='S' && pvr[2]=='S') ||
            (pvr[3]=='S' && pvr[2]=='T') ||
            (pvr[3]=='T' && pvr[2]=='M') ||
            (pvr[3]=='U' && pvr[2]=='I') ||
            (pvr[3]=='U' && pvr[2]=='L') ||
            (pvr[3]=='U' && pvr[2]=='S') )
          {
            // These explicit VR's have a 16-bit length. This
            // allows them to fit into the same space as implicit VR.
            explicitVR = 1;
            e_len = 256 * pvr[0] + pvr[1];
          }

          // Do we still have the byte ordering problem?
          if (e_len == 0x04000000)
          {
            vvDebugMsg::msg(2, "\nPossible byte ordering problem - switching!");
            info->littleEndian = !info->littleEndian;
            fseek(_fp, where, 0);
            group   = read16(_fp);
            element = read16(_fp);
            read32(_fp);
            e_len = 4;                            // don't do e_len conversion again
          }
        }
      }
    }
    else                                          // this isn't the first one so check the value representation
    {
      if (explicitVR)
      {
        char* pvr = (char*)&e_len;
        if      ( (pvr[0]=='O' && pvr[1]=='B') ||
          (pvr[0]=='O' && pvr[1]=='W') ||
          (pvr[0]=='S' && pvr[1]=='Q') )
        {
          // these explicit vr's have a 32-bit length.
          // therefore, we'll need to read another 16-bits &
          // then include these addition 16-bits in the
          // current 16-bit length.
          e_len = read32(_fp);
        }
        else if ( (pvr[3]=='O' && pvr[2]=='B') ||
          (pvr[3]=='O' && pvr[2]=='W') ||
          (pvr[3]=='S' && pvr[2]=='Q') )
        {
          // these explicit vr's have a 32-bit length.
          // therefore, we'll need to read another 16-bits &
          // then include these addition 16-bits in the
          // current 16-bit length.
          e_len = read32(_fp);
        }
        else if ( (pvr[0]=='A' && pvr[1]=='E') ||
          (pvr[0]=='A' && pvr[1]=='S') ||
          (pvr[0]=='A' && pvr[1]=='T') ||
          (pvr[0]=='C' && pvr[1]=='S') ||
          (pvr[0]=='D' && pvr[1]=='A') ||
          (pvr[0]=='D' && pvr[1]=='S') ||
          (pvr[0]=='D' && pvr[1]=='T') ||
          (pvr[0]=='F' && pvr[1]=='L') ||
          (pvr[0]=='F' && pvr[1]=='D') ||
          (pvr[0]=='I' && pvr[1]=='S') ||
          (pvr[0]=='L' && pvr[1]=='O') ||
          (pvr[0]=='L' && pvr[1]=='T') ||
          (pvr[0]=='P' && pvr[1]=='N') ||
          (pvr[0]=='S' && pvr[1]=='H') ||
          (pvr[0]=='S' && pvr[1]=='L') ||
          (pvr[0]=='S' && pvr[1]=='S') ||
          (pvr[0]=='S' && pvr[1]=='T') ||
          (pvr[0]=='T' && pvr[1]=='M') ||
          (pvr[0]=='U' && pvr[1]=='I') ||
          (pvr[0]=='U' && pvr[1]=='L') ||
          (pvr[0]=='U' && pvr[1]=='S') )
        {
          // These explicit VR's have a 16-bit length. This
          // allows them to fit into the same space as implicit VR.
          e_len &= 0xffff0000;
          e_len >>= 16;
        }
        else if ( (pvr[3]=='A' && pvr[2]=='E') ||
          (pvr[3]=='A' && pvr[2]=='S') ||
          (pvr[3]=='A' && pvr[2]=='T') ||
          (pvr[3]=='C' && pvr[2]=='S') ||
          (pvr[3]=='D' && pvr[2]=='A') ||
          (pvr[3]=='D' && pvr[2]=='S') ||
          (pvr[3]=='D' && pvr[2]=='T') ||
          (pvr[3]=='F' && pvr[2]=='L') ||
          (pvr[3]=='F' && pvr[2]=='D') ||
          (pvr[3]=='I' && pvr[2]=='S') ||
          (pvr[3]=='L' && pvr[2]=='O') ||
          (pvr[3]=='L' && pvr[2]=='T') ||
          (pvr[3]=='P' && pvr[2]=='N') ||
          (pvr[3]=='S' && pvr[2]=='H') ||
          (pvr[3]=='S' && pvr[2]=='L') ||
          (pvr[3]=='S' && pvr[2]=='S') ||
          (pvr[3]=='S' && pvr[2]=='T') ||
          (pvr[3]=='T' && pvr[2]=='M') ||
          (pvr[3]=='U' && pvr[2]=='I') ||
          (pvr[3]=='U' && pvr[2]=='L') ||
          (pvr[3]=='U' && pvr[2]=='S') )
        {
          // These explicit VR's have a 16-bit length. This
          // allows them to fit into the same space as implicit VR.
          e_len = 256*pvr[0] + pvr[1];
        }
        else if ( (pvr[0]=='U' && pvr[1]=='N') )
        {
          // this is unknown, the length is in the next four bytes;
          e_len = read32(_fp);
        }
      }
    }

    switch (group)
    {
      case 0x0002:                                // group 2
        switch (element)
        {
          case 0x00 :  infoText = "file meta elements group len"; break;
          case 0x01 :  infoText = "file meta info version";       break;
          case 0x02 :  infoText = "media storage SOP class uid";  break;
          case 0x03 :  infoText = "media storage SOP inst uid";   break;
          case 0x10 :  infoText = "transfer syntax uid";          break;
          case 0x12 :  infoText = "implementation class uid";     break;
          case 0x13 :  infoText = "implementation version name";  break;
          case 0x16 :  infoText = "source app entity title";      break;
          case 0x100:  infoText = "private info creator uid";     break;
          case 0x102:  infoText = "private info";                 break;
        }
        break;

      case 0x0008:                                // group 8
        switch (element)
        {
          case 0x00 :  infoText = "identifying group";             break;
          case 0x01 :  infoText = "length to end";                 break;
          case 0x05 :  infoText = "specific character set"; t=VV_STRING; break;
          case 0x08 :  infoText = "image type";       t=VV_STRING; break;
          case 0x10 :  infoText = "recognition code";              break;
          case 0x12 :  infoText = "Instance Creation Date"; t=VV_STRING;   break;
          case 0x13 :  infoText = "Instance Creation Time"; t=VV_STRING;   break;
          case 0x14 :  infoText = "Instance Creator UID";              break;
          case 0x16 :  infoText = "SOP Class UID";    t=VV_STRING; break;
          case 0x18 :  infoText = "SOP Instance UID";              break;
          case 0x20 :  infoText = "study date";       t=VV_STRING; break;
          case 0x21 :  infoText = "series date";      t=VV_STRING; break;
          case 0x22 :  infoText = "acquisition date"; t=VV_STRING; break;
          case 0x23 :  infoText = "image date";       t=VV_STRING; break;
          case 0x30 :  infoText = "study time";                    break;
          case 0x31 :  infoText = "series time";                   break;
          case 0x32 :  infoText = "acquisition time";              break;
          case 0x33 :  infoText = "image time";                    break;
          case 0x40 :  infoText = "data set type";                 break;
          case 0x41 :  infoText = "data set subtype";              break;
          case 0x50 :  infoText = "accession number";              break;
          case 0x60 :  infoText = "modality";         t=VV_STRING; break;
          case 0x70 :  infoText = "manufacturer";     t=VV_STRING; break;
          case 0x80 :  infoText = "institution name"; t=VV_STRING; break;
          case 0x90 :  infoText = "referring physician's name";    break;
          case 0x1010: infoText = "station name";                  break;
          case 0x103e: infoText = "series description";            break;
          case 0x1030: infoText = "study description";             break;
          case 0x1040: infoText = "institutional dept. name";      break;
          case 0x1050: infoText = "performing physician's name"; t=VV_STRING; break;
          case 0x1060: infoText = "name phys(s) read stdy";        break;
          case 0x1070: infoText = "operator's name";  t=VV_STRING; break;
          case 0x1080: infoText = "admitting diagnoses description"; t=VV_STRING; break;
          case 0x1090: infoText = "manufacturer's model name"; t=VV_STRING; break;
          case 0x1140: infoText = "referenced image sequence"; t=VV_STRING; break;
        }
        break;

      case 0x0009:                                // group 9
        switch (element)
        {
          case 0x10:   t=VV_STRING; break;
          case 0x12:   t=VV_STRING; break;
          case 0x13:   t=VV_STRING; break;
          case 0x1010: t=VV_STRING; break;
          case 0x1015: t=VV_STRING; break;
          case 0x1040: break;
          case 0x1041: t=VV_STRING; break;
          case 0x1210: t=VV_STRING; break;
          case 0x1212: break;
          case 0x1213: break;
          case 0x1214: break;
          case 0x1226: t=VV_STRING; break;
          case 0x1227: break;
          case 0x1316: break;
          case 0x1320: break;
        }
        break;

      case 0x0010:                                // group 10h
        switch (element)
        {
          case 0x00 :  infoText = "patient group";                           break;
          case 0x10 :  infoText = "patient name";               t=VV_STRING; break;
          case 0x20 :  infoText = "patient ID";                 t=VV_STRING; break;
          case 0x30 :  infoText = "patient birthdate";          t=VV_STRING; break;
          case 0x40 :  infoText = "patient sex";                t=VV_STRING; break;
          case 0x1010: infoText = "patient age";                t=VV_STRING; break;
          case 0x1030: infoText = "patient weight";             t=VV_STRING; break;
          case 0x21b0: infoText = "additional patient history"; t=VV_STRING; break;
        }
        break;

      case 0x0018:                                // group 18h
        switch (element)
        {
          case 0x00 :  infoText = "acquisition group";             break;
          case 0x10 :  infoText = "contrast/bolus agent";    t=VV_STRING; break;
          case 0x15 :  infoText = "body part examined";      t=VV_STRING; break;
          case 0x20 :  infoText = "scanning sequence";       t=VV_STRING; break;
          case 0x21 :  infoText = "sequence variant";        t=VV_STRING; break;
          case 0x22 :  infoText = "scan options";            t=VV_STRING; break;
          case 0x23 :  infoText = "MR acquisition type";     t=VV_STRING; break;
          case 0x24 :  infoText = "sequence name";           t=VV_STRING; break;
          case 0x25 :  infoText = "angio flag";              t=VV_STRING; break;
          case 0x30 :  infoText = "radionuclide";            t=VV_STRING; break;
          case 0x50 :  infoText = "slice thickness";
          {
            t=VV_STRING;
            char* buff = new char[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n==1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = '\0';
            float d = 0.0f;
            int ret = sscanf(buff, "%f", &d);
            if (ret != 1)
            {
              cerr << "vvDicom::readDicomData: read error" << endl;
            }
            if (d > 0.0f) info->sliceThickness = d;
            delete[] buff;
          }
          break;
          case 0x60:   infoText = "kvp";                     t=VV_STRING; break;
          case 0x80 :  infoText = "repetition time";         t=VV_STRING; break;
          case 0x81 :  infoText = "echo time";               t=VV_STRING; break;
          case 0x82 :  infoText = "inversion time";          t=VV_STRING; break;
          case 0x83 :  infoText = "number of averages";      t=VV_STRING; break;
          case 0x84 :  infoText = "imaging frequency";       t=VV_STRING; break;
          case 0x85 :  infoText = "imaged nucleus";          t=VV_STRING; break;
          case 0x86 :  infoText = "echo number";             t=VV_STRING; break;
          case 0x87 :  infoText = "magnetic field strength"; t=VV_STRING; break;
          case 0x88 :  infoText = "spacing between slices";
          {
            t=VV_STRING;
            char* buff = new char[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n==1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = '\0';
            float d = 0.0f;
            int ret = sscanf(buff, "%f", &d);
            if (ret != 1)
            {
              cerr << "vvDicom::readDicomData: read error" << endl;
            }
            if (d > 0.0f) info->dist[2] = d;
            delete[] buff;
          }
          break;
          case 0x91 :  infoText = "echo train length";       t=VV_STRING; break;
          case 0x93 :  infoText = "percent sampling";        t=VV_STRING; break;
          case 0x94 :  infoText = "percent phase field of view"; t=VV_STRING; break;
          case 0x95 :  infoText = "pixel bandwidth";         t=VV_STRING; break;
          case 0x1020: infoText = "software version(s)";     t=VV_STRING; break;
          case 0x1030: infoText = "protocol name";           t=VV_STRING; break;
          case 0x1040: infoText = "contrast/Bolus route";    t=VV_STRING; break;
          case 0x1062: infoText = "nominal interval";        t=VV_STRING; break;
          case 0x1088: infoText = "heart rate";              t=VV_STRING; break;
          case 0x1090: infoText = "cardiac number of images"; t=VV_STRING; break;
          case 0x1094: infoText = "trigger window";          t=VV_STRING; break;
          case 0x1100: infoText = "reconstruction diameter"; t=VV_STRING; break;
          case 0x1120: infoText = "gantry/detector tilt";    t=VV_STRING; break;
          case 0x1150: infoText = "exposure time";           t=VV_STRING; break;
          case 0x1151: infoText = "x-ray tube current";      t=VV_STRING; break;
          case 0x1210: infoText = "convolution kernel";      t=VV_STRING; break;
          case 0x1250: infoText = "receiving coil";          t=VV_STRING; break;
          case 0x1251: infoText = "transmitting coil";       t=VV_STRING; break;
          case 0x1310: infoText = "acquisition matrix";      t=VV_STRING; break;
          case 0x1314: infoText = "flip angle";              t=VV_STRING; break;
          case 0x1315: infoText = "variable flip angle flag"; t=VV_STRING; break;
          case 0x1316: infoText = "SAR";                     t=VV_STRING; break;
          case 0x1400: infoText = "acquisition device processing description"; t=VV_STRING; break;
          case 0x1401: infoText = "acquisition device processing code"; t=VV_STRING; break;
          case 0x5100: infoText = "patient position";        t=VV_STRING; break;
          case 0x5101: infoText = "view position";           t=VV_STRING; break;
          case 0x6000: infoText = "sensitivity";             t=VV_STRING; break;
        }
        break;

      case 0x0020:                                // group 20h
        switch (element)
        {
          case 0x00 :  infoText = "relationship group";            break;
          case 0x0d :  infoText = "study instance UID";      t=VV_STRING; break;
          case 0x0e :  infoText = "series instance UID";     t=VV_STRING; break;
          case 0x10 :  infoText = "study ID";                t=VV_STRING; break;
          case 0x11 :  infoText = "series number";
          t=VV_STRING;
          {
            char* buff = new char[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n==1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = '\0';
            int num;
            int ret = sscanf(buff, "%d", &num);
            if (ret != 1)
            {
              cerr << "vvDicom::readDicomData: read error" << endl;
            }
            info->sequence = num;
            delete[] buff;
          }
          break;
          case 0x12 :  infoText = "acquisition number"; t = VV_STRING; break;
          case 0x13 :  infoText = "image number";
          t=VV_STRING;
          {
            char* buff = new char[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n==1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = '\0';
            int num;
            int ret = sscanf(buff, "%d", &num);
            if (ret != 1)
            {
              cerr << "vvDicom::readDicomData: read error" << endl;
            }
            info->image = num;
            delete[] buff;
          }
          break;
          case 0x20 :  infoText = "patient orientation"; t=VV_STRING; break;
          case 0x30 :  infoText = "image position (ret)";          break;
          case 0x32 :  infoText = "image position (patient)"; t=VV_STRING; break;
          case 0x35 :  infoText = "image orientation (ret)";       break;
          case 0x37 :  infoText = "image orientation (patient)"; t=VV_STRING; break;
          case 0x50 :  infoText = "location (ret)";                break;
          case 0x52 :  infoText = "frame of reference UID"; t=VV_STRING; break;
          case 0x60 :  infoText = "laterality";                    break;
          case 0x1002: infoText = "images in acquisition";         break;
          case 0x1040: infoText = "position reference";            break;
          case 0x1041: infoText = "slice location";
          {
            t=VV_STRING;
            char* buff = new char[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n==1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = '\0';
            int ret = sscanf(buff, "%f", &info->slicePos);
            if (ret != 1)
            {
              cerr << "vvDicom::readDicomData: read error" << endl;
            }
            delete[] buff;
          }
          break;
          case 0x3401: infoText = "modifying device id";           break;
          case 0x3402: infoText = "modified image id";             break;
          case 0x3403: infoText = "modified image date";           break;
          case 0x3404: infoText = "modifying device mfg.";         break;
          case 0x3405: infoText = "modified image time";           break;
          case 0x3406: infoText = "modified image desc.";          break;
          case 0x4000: infoText = "image comments";   t=VV_STRING; break;
          case 0x5000: infoText = "original image id";             break;
        }
        break;

      case 0x0028:                                // group 28h
        switch (element)
        {
          case 0x00 :  infoText = "image presentation group";      break;
          case 0x02 :  infoText = "samples per pixel";
          t=VV_STRING;
          {
            char* buff = (char*)new uchar[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = '\0';
            if (e_len >= 1)
              info->chan = static_cast<int>(buff[0]);
            else
              cerr << "vvDicom::readDicomData: read error for (0028,0002) Samples Per Pixel" << endl;
            delete[] buff;
          }
          break;
          case 0x04 :  infoText = "photometric interpretation";
          t=VV_STRING;
          {
            char* buff = (char*)new uchar[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = 0;
            if (strncmp(buff,"MONOCHROME1",strlen("MONOCHROME1")) == 0)
              photometric_specified = 1;
            else if (strncmp(buff,"MONOCHROME2",strlen("MONOCHROME2")) == 0)
              photometric_specified = 2;
            else if (strncmp(buff,"RGB",strlen("RGB")) == 0)
              photometric_specified = 4;
            delete[] buff;
          }
          break;
          case 0x05 :  infoText = "image dimensions (ret)";        break;
          case 0x06 :  infoText = "planar configuration";          break;
          case 0x08 :  infoText = "number of frames";              break;
          case 0x09 :  infoText = "frame increment pointer";       break;
          case 0x10 :  infoText = "rows";
          assert(e_len == 2);
          pos = ftell(_fp);
          info->height = read16(_fp);
          fseek(_fp, pos, SEEK_SET);
          break;
          case 0x11 :  infoText = "columns";
          assert(e_len == 2);
          pos = ftell(_fp);
          info->width = read16(_fp);
          fseek(_fp, pos, SEEK_SET);
          break;
          case 0x30 :  infoText = "pixel spacing";
          {
            char* buff = new char[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n==1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = '\0';
            float a=0.0f, b=0.0f;
            int ret = sscanf(buff, "%f\\%f", &a, &b);
            if (ret != 2)
            {
              cerr << "vvDicom::readDicomData: read error" << endl;
            }
            if (a > 0.0f && b > 0.0f)
            {
              info->dist[0] = a;
              info->dist[1] = b;
            }
            delete[] buff;
          }
          break;
          case 0x31 :  infoText = "zoom factor";                   break;
          case 0x32 :  infoText = "zoom center";                   break;
          case 0x34 :  infoText = "pixel aspect ratio";            break;
          case 0x40 :  infoText = "image format (ret)";            break;
          case 0x50 :  infoText = "manipulated image (ret)";       break;
          case 0x51 :  infoText = "corrected image";               break;
          case 0x60 :  infoText = "compression code (ret)";        break;
          case 0x0100: infoText = "bits allocated";
          assert(e_len == 2);
          pos = ftell(_fp);
          tmp = read16(_fp);
          fseek(_fp, pos, SEEK_SET);
          if (tmp==8) info->bpp = 1;
          else if (tmp==16) info->bpp = 2;
          else
          {
            cerr << "DICOM data must be 8 or 16 bit per pixel. Cannot read " << tmp << " bit." << endl;
            assert(0);
          }
          break;
          case 0x0101: infoText = "bits stored";
          assert(e_len == 2);
          pos = ftell(_fp);
          info->bitsStored = read16(_fp);
          fseek(_fp, pos, SEEK_SET);
          break;
          case 0x0102: infoText = "high bit";
          assert(e_len == 2);
          pos = ftell(_fp);
          info->highBit = read16(_fp);
          fseek(_fp, pos, SEEK_SET);
          break;
          case 0x0103: infoText = "pixel representation";
          assert(e_len == 2);
          pos = ftell(_fp);
          tmp = read16(_fp);
          fseek(_fp, pos, SEEK_SET);
          if      (tmp == 0)  info->isSigned = false;
          else if (tmp == 1)  info->isSigned = true;
          break;
          case 0x0200: infoText = "image location (ret)"; break;
          case 0x1050: infoText = "window center";
          t = VV_FLOAT;
          window_center_specified = 1;
          {
            char* buff = (char*)new uchar[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = 0;
            char* endptr;
            window_center = strtod(buff, &endptr);
            // Check for a conversion error:
            if (endptr == buff) window_center_specified = 0;
            delete[] buff;
          }
          break;
          case 0x1051: infoText = "window width";
          t = VV_FLOAT;
          window_width_specified = 1;
          {
            char* buff = (char*)new uchar[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = 0;
            char* endptr;
            window_width = strtod(buff, &endptr);
            // Check for a conversion error:
            if (endptr == buff) window_width_specified = 0;
            delete[] buff;
          }
          break;

          case 0x1052: infoText = "rescale intercept";
          t = VV_FLOAT;
          intercept_specified = 1;
          {
            char* buff = (char*)new uchar[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            fseek(_fp, pos, SEEK_SET);
            assert(n == 1);
            buff[e_len] = 0;
            char* endptr;
            intercept = strtod(buff, &endptr);
            // Check for a conversion error:
            if (endptr == buff)  intercept_specified = 0;
            delete[] buff;
          }
          break;
          case 0x1053: infoText = "rescale slope";
          t = VV_FLOAT;
          slope_specified = 1;
          {
            char* buff = (char*)new uchar[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = 0;
            char* endptr;
            slope = strtod(buff, &endptr);
            // Check for a conversion error:
            if (endptr == buff)  slope_specified = 0;
            delete[] buff;
          }
          break;
          case 0x1100: infoText = "gray lookup table desc (ret)";  break;
          case 0x1200: infoText = "gray lookup table data (ret)";  break;
          case 0x1201: infoText = "red table";
          red_table_size = e_len;
          red_table = (ushort*)new uchar[e_len * sizeof(*red_table)];
          {
            pos = ftell(_fp);
            size_t n = fread(red_table, e_len*sizeof(*red_table),1,_fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
          }
          flip16bitData ((uchar*)red_table, e_len);
          break;
          case 0x1202: infoText = "green table";
          green_table_size = e_len;
          green_table = (ushort*)new uchar[e_len * sizeof(*green_table)];
          {
            pos = ftell(_fp);
            size_t n = fread(green_table, e_len*sizeof(*green_table),1,_fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
          }
          break;
          case 0x1203: infoText = "blue table";
          blue_table_size = e_len;
          blue_table = (ushort*)new uchar[e_len * sizeof(*blue_table)];
          {
            pos = ftell(_fp);
            size_t n = fread(blue_table, e_len*sizeof(*blue_table),1,_fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
          }
          break ;
          case 0x2114 :  infoText = "Lossy Image Compression Method";
          t=VV_STRING;
          {
            char* buff = (char*)new uchar[e_len+1];
            pos = ftell(_fp);
            size_t n = fread(buff, e_len, 1, _fp);
            if(n != 1)
              cerr << "Returned value from fread is " << n << ", expected 1" << endl;
            assert(n == 1);
            fseek(_fp, pos, SEEK_SET);
            buff[e_len] = 0;
            fprintf(stderr,"lossy compression %s\n",buff);
            delete[] buff;
          }
          break;
        }
        break;

      case 0x4000:                                // group 4000h
        infoText = "text";
        break;

      case 0x7FE0:                                // group 7fe0h
        switch (element)
        {
          case 0x00: infoText = "pixel data";              break;
          case 0x10: infoText = "pixel data"; done = true; break;
        }
        break;

      default:                                    // other group
        if (group >= 0x6000 && group <= 0x601e && (group & 1) == 0) infoText = "overlay";
        if (element == 0x0000)  infoText = "group length";
        if (element == 0x4000)  infoText = "comments";
        break;
    }

    if (vvDebugMsg::isActive(2))
    {
      fprintf(stderr, "%04lx:%04x,%04x,%02d", where, group, element, e_len);
    }

    if (infoText == NULL)
    {
      if (t==VV_STRING) infoText = "unrecognized text";
      else infoText = "unrecognized";
    }
    if (vvDebugMsg::isActive(2))
    {
      fprintf(stderr, " %26s: ", infoText);
    }

    if (!done)
    {
      char* endptr;
      int i;
      char* buff;

      buff = (char*)new uchar[e_len + 1];
      if (e_len > 0)
      {
        size_t n = fread(buff, e_len, 1, _fp);
        if (n!=1) cerr << "\nError: e_len = " << e_len << endl;
        assert(n == 1);
      }
      // Display element data depending on data type:
      switch (t)
      {
        case VV_STRING:                           // display entire string
          if (vvDebugMsg::isActive(2))
          {
            for (i = 0; i < e_len; i++)
            {
              fprintf(stderr, "%c", buff[i]);
            }
          }
          break;
        case VV_FLOAT:
          buff[e_len] = 0;
          if (vvDebugMsg::isActive(2))
          {
            fprintf(stderr, "%f", strtod(buff, &endptr));
          }
          break;
        case VV_UNKNOWN:
          switch (e_len)
          {
            case 1 :                              // display 8 bit value
              if (vvDebugMsg::isActive(2))
              {
                fprintf(stderr, "%u", (unsigned int) buff[0]);
              }
              break;
            case 2 :                              // display 16 bit value
              if (info->littleEndian)
                i = buff[0] + 256*buff[1];
              else
                i = buff[0]*256 + buff[1];
              if (vvDebugMsg::isActive(2))
              {
                fprintf(stderr, "%d", i);
              }
              break;
            case 4 :                              // display 32 bit value
              if (info->littleEndian)
                i = buff[0] + 256*buff[1] + 256*256*buff[2] +
                  256*256*256*buff[3];
              else
                i = buff[0]*256*256*256 + buff[1]*256*256 +
                  buff[2]*256 + buff[3];
              if (vvDebugMsg::isActive(2))
              {
                fprintf(stderr, "%d", i);
              }
              break;
            default:                              // don't display other elements
              break;
          }
          break;
      }
      delete[] buff;
    }
    if (vvDebugMsg::isActive(2))
    {
      cerr << endl;
    }
  }

  // Read the actual pixel data:
  info->raw = new uchar[info->height * info->width * info->bpp * info->chan];
  size_t n = fread(info->raw,  1, info->height * info->width * info->bpp * info->chan, _fp);
  //if(n != 1)
    //cerr << "vvDicom::readDicomData: Error reading actual pixel data" << endl;
    cerr << "vvDicom::readDicomData:  reading actual pixel data " <<n<< endl;
}

//----------------------------------------------------------------------------
/** This function flips i.e. reverses the 'endian-ness' of the input data
  (when the input is a 1d array).
*/
void vvDicom::flip16bitData(uchar* buff, int w)
{
  assert(buff != NULL);
  for (int i = 0; i < 2*w; i+=2)
  {
    uchar tmp = buff[i];
    buff[i]   = buff[i+1];
    buff[i+1] = tmp;
  }
}

//----------------------------------------------------------------------------
void vvDicom::handlePhotometricInterpretation()
{
  short* tmp16;
  long maxValue;
  long i;

  assert(info->bpp==1 || info->bpp==2);
  if (info->raw==NULL || info->width <= 0 || info->height <= 0) return;
  if (photometric_specified == 4 && info->chan != 3)
    cerr << "Inconsistency between (0028,0002) Samples Per Pixel and (0028,0004) Photometric Interpretation" << endl;
  if (photometric_specified != 1)  return;
  maxValue = 1 << info->bpp;
  assert(maxValue == (1 << info->bpp));
  tmp16 = (short*)info->raw;
  for (i=0; i<info->width*info->height; i++)
  {
    if (info->bpp==1)
      info->raw[i] = (char)(maxValue - info->raw[i]);
    else
      tmp16[i] = (short)(maxValue - tmp16[i]);
  }
}

//----------------------------------------------------------------------------
/** Main routine to read a DICOM file.
  @param fname file name
  @param prop DICOM properties, will be filled by this method
  @return true if ok, false on error
*/
bool vvDicom::readDicomFile(char* fname)
{
  assert(fname != NULL);

  setEndian();
  fp = fopen(fname, "rb");
  if (fp==NULL) return false;
  readDicomData(fp);
  handlePhotometricInterpretation();

  fclose(fp);
  fp = NULL;

  return true;
}

vvDicomProperties::vvDicomProperties()
{
  raw = NULL;
  width = height = 0;
  bpp = 2;
  chan = 1;
  sequence = 0;
  image = 0;
  bitsStored = 16;
  highBit = 0;
  isSigned = false;
  littleEndian = false;
  dist[0] = dist[1] = dist[2] = 1.0f;
  slicePos = 0.0f;
}

void vvDicomProperties::print()
{
  cerr << "DICOM Info:" << endl;
  cerr << "width:           " << width << endl;
  cerr << "height:          " << height << endl;
  cerr << "sequence:        " << sequence << endl;
  cerr << "image number:    " << image << endl;
  cerr << "bytes allocated: " << bpp << endl;
  cerr << "bits stored:     " << bitsStored << endl;
  cerr << "high bit:        " << highBit << endl;
  cerr << "signed:          " << int(isSigned) << endl;
  cerr << "little endian:   " << int(littleEndian) << endl;
  cerr << "voxel distance:  " << dist[0] << ", " << dist[1] << ", " << dist[2] << endl;
  cerr << "slice location:  " << slicePos << endl;
}

// EOF
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
