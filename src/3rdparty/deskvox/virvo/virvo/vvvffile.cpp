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
#include <math.h>
#include "vvdebugmsg.h"
#include "vvvffile.h"

using namespace std;

vvvffile::vvvffile(char* fName)
{
  errnmb = 0;
  FILEREAD = 0;
  channel_array_length = 1;
  icontype = 0;
  time_stamp = 0;
  number_of_pins = 0;
  number_of_pin_lists = 0;
  fileName = new char[strlen(fName) + 1];
  strcpy(fileName, fName);

  // noch keine Beschreibung vorhanden
  descr = NULL;
  channel_description = NULL;
  icondata = NULL;
  tempfile = NULL;
  channel = NULL;
  pin_data = NULL;

  return;
}

vvvffile::~vvvffile(void)
{
  delete[] fileName;
  if (descr)
    delete[] descr;
  if (icondata)
    delete[] icondata;
  if (tempfile)
  {
    delete[] tempfile;
    tempfile = NULL;
  }

  if (channel_description)
    delete[] channel_description;
  if (channel)
    delete[] channel;
  if (pin_data)
    delete[] pin_data;
  if (number_of_pins)
    delete[] number_of_pins;

  return;
}

int vvvffile::readFile(void)
{
  unsigned char a3, a4, a5, a6, a7, a8;
  int*  channel_types;
  int   jump_width;

  // everything ok, till now :)
  errnmb = 0;
  proceeded = 0;

  // Open file
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: open file for LOADing: " << fileName << "		[ OK ]" << endl;

  // Check identity
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  a5 = (unsigned char)fgetc(dz);
  a6 = (unsigned char)fgetc(dz);
  a7 = (unsigned char)fgetc(dz);
  a8 = (unsigned char)fgetc(dz);

  if (a1 != 86 || a2 != 70 || a3 != 45 || a4 != 70 || a5 != 105 || a6 != 108 || a7 != 101 || a8 != 32)
  {
    cerr << "## Error: file corrupt or not a volume data file		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Identity checked: VF-File					[ OK ]" << endl;

  checksum = 0;
  proceeded = 8;

  // Check version-number
  if (fgetc(dz) != 118)
  {
    cerr << "## Error in File-Header or wrong Version			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  version_number = fgetc(dz) - 48;
  if ((version_number < 0) || (version_number > 9))
  {
    cerr << "## Error in File-Header or wrong Version			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  if (fgetc(dz) != 46)
  {
    cerr << "## Error in File-Header or wrong Version			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  version_number = version_number * 10 + (fgetc(dz) - 48);
  version_number = version_number * 10 + (fgetc(dz) - 48);
  if (version_number < 1 || version_number > 300)
  {
    cerr << "## Error in File-Header or wrong Version			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  if (vvDebugMsg::isActive(3))
  {
    cerr << "readFile: Version checked: v" << int(version_number / 100) << "." << int(version_number / 10) % 10 << version_number % 10 << "					[ OK ]" << endl;
  }

  checksum += 0;
  proceeded += 5;

  // resolve headersize
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  headersize = octals2int(0, 0, a1, a2);
  if (headersize < 30)
  {
    cerr << "## Error: file corrupt or not a volume data file		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Headersize checked: " << headersize << "					[ OK ]" << endl;

  checksum += a1 + a2;
  proceeded += 2;

  // resolve number of data-arrays
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  number_of_data_arrays = octals2int(0, 0, a1, a2);
  if (number_of_data_arrays < 1)
  {
    cerr << "## Error: impossible number of data arrays given: " << number_of_data_arrays << "			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Number of Data-Arrays checked: " << number_of_data_arrays << "				[ OK ]" << endl;

  checksum += a1 + a2;
  proceeded += 2;

  // resolve volume dimensions
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  voxel_number_x = octals2int(a1, a2, (unsigned char)a3, (unsigned char)a4);
  checksum += a1 + a2 + a3 + a4;
  proceeded += 4;
  if (voxel_number_x == 0)
  {
    cerr << "## Error: file corrupt or not a volume data file		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Voxel Number X checked: " << voxel_number_x << "					[ OK ]" << endl;

  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  voxel_number_y = octals2int(a1, a2, a3, a4);
  checksum += a1 + a2 + a3 + a4;
  proceeded += 4;
  if (voxel_number_y == 0)
  {
    cerr << "## Error: file corrupt or not a volume data file	[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Voxel Number Y checked: " << voxel_number_y << "					[ OK ]" << endl;

  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  voxel_number_z = octals2int(a1, a2, a3, a4);
  checksum += a1 + a2 + a3 + a4;
  proceeded += 4;
  if (voxel_number_z == 0)
  {
    cerr << "## Error: file corrupt or not a volume data file	[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Voxel Number Z checked: " << voxel_number_z << "					[ OK ]" << endl;

  // resolve number of bytes every voxel is described with
  a1 = (unsigned char)fgetc(dz);
  bytes_per_voxel = a1;
  checksum += a1;
  proceeded += 1;
  if (bytes_per_voxel == 0)
  {
    cerr << "## Error: file corrupt or not a volume data file	[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Bytes per Voxel checked: " << bytes_per_voxel << "					[ OK ]" << endl;

  // Jump over unassigned space
  if (headersize != proceeded)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "readFile: unused extensions: " << int(headersize - proceeded) << " bytes				[ OK ]" << endl;
    while (proceeded < headersize)
    {
      a1 = (unsigned char)fgetc(dz);
      checksum += a1;
      proceeded += 1;
    }
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: no extensions used						[ OK ]" << endl;

  // testing checksum
  checksum = checksum % 256;
  a1 = (unsigned char)fgetc(dz);
  if (int(a1) != checksum)
  {
    cerr << "## Error: checksum test failed				[ FAILED ]" << endl;
    cerr << "Found " << int(a1) << " is supposed to be " << checksum << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Checksum checked: " << checksum << "						[ OK ]" << endl;

  if (vvDebugMsg::isActive(3))
    cerr << "readFile: Header completely loaded:					[ OK ]" << endl;

  // resolve icon related space and jump over it
  // which type of icon ...
  icontype = fgetc(dz);
  switch (icontype)
  {
    case 0:
      if (vvDebugMsg::isActive(3))
        cerr << "readFile: Icon-Type checked: no icon					[ OK ]" << endl;
      iconsize = 0;
      break;
    case 1:
      if (vvDebugMsg::isActive(3))
        cerr << "readFile: Icon-Type checked: standard					[ OK ]" << endl;

      // 'standard' icon = bitmap, 32x32, rgb 24bit)
      fseek(dz, 3072L, SEEK_CUR);
      iconsize = 3072;
      break;
    case 2:
      if (vvDebugMsg::isActive(3))
        cerr << "readFile: Icon-Type checked: standard					[ OK ]" << endl;
      fseek(dz, 64L * 64L * 3L, SEEK_CUR);
      iconsize = 64 * 64 * 3;
      break;
    default:
      cerr << "## Error: unrecognized icon-type " << icontype << "	[ FAILED ]" << endl;
      iconsize = 0;
      errnmb = 1;
      return errnmb;

      // break; // unreachable code
  }

  // read included description text
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  textlength = octals2int(0, 0, a1, a2);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: Description text checked: " << textlength << "					[ OK ]" << endl;

  // jump over text
  fseek(dz, int(textlength), SEEK_CUR);

  // read Checkbit - end of header, start of data
  a1 = (unsigned char)fgetc(dz);
  if (a1 != 66)
  {
    cerr << "## Error: Checkbit #1 is wrong			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Checkbit #1 reached: 66					[ OK ]" << endl;

  // jump over data
  for (i = 0; i < number_of_data_arrays; i++)
  {
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    a3 = (unsigned char)fgetc(dz);
    a4 = (unsigned char)fgetc(dz);
    data_array_length = octals2int(a1, a2, a3, a4);
    fseek(dz, int(data_array_length), SEEK_CUR);
  }

  if (vvDebugMsg::isActive(3))
  {
    cerr << "readFile: Jumped over data: " << number_of_data_arrays << " array(s), " << int(data_array_length) << " bytes			[ OK ]" << endl;
  }

  // read Checkbit - end of data, start of tail
  a1 = (unsigned char)fgetc(dz);
  if (a1 != 99)
  {
    cerr << "## Error: Checkbit #2 is wrong ... not the best sign	 [ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readFile: Checkbit #2 reached: 99					[ OK ]" << endl;

  // read tailsize
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  tailsize = octals2int(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve tailsize: " << tailsize << "  					[ OK ]" << endl;

  // read x-coordinate of hotspot
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  hotspot_x = octals2int(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve hotspot_x: " << hotspot_x << "  					[ OK ]" << endl;

  // read y-coordinate of hotspot
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  hotspot_y = octals2int(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve hotspot_y: " << hotspot_y << "  					[ OK ]" << endl;

  // read z-coordinate of hotspot
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  hotspot_z = octals2int(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve hotspot_z: " << hotspot_z << "  					[ OK ]" << endl;

  // read real voxel dimensions (in x-direction)
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  realsize_x = octals2float(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve realsize_x: " << realsize_x << "       				[ OK ]" << endl;

  // read real voxel dimensions (in y-direction)
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  realsize_y = octals2float(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve realsize_y: " << realsize_y << "       				[ OK ]" << endl;

  // read real voxel dimensions (in z-direction)
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  realsize_z = octals2float(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve realsize_z: " << realsize_z << "       				[ OK ]" << endl;

  // read number of channels
  a1 = (unsigned char)fgetc(dz);
  number_of_channels = a1;
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: resolve number of channels: " << number_of_channels << " 				[ OK ]" << endl;

  // jump over channels
  /*   fseek( dz, int(tailsize-29), SEEK_CUR );
     if(vvDebugMsg::isActive(3))
     cerr << "readFile: jumping over channels: " << int(tailsize-29) << " bytes				[ OK ]" << endl;
  */
  channel_types = new int[number_of_channels];
  i = 0;
  for (j = 0; j < number_of_channels; j++)
  {
    a1 = (unsigned char)fgetc(dz);

    //cerr << endl << endl;
    //cerr << "Text-Length: " << int(a1) << endl;
    fseek(dz, int(a1), SEEK_CUR);                 // description text
    a1 = (unsigned char)fgetc(dz);
    channel_types[i] = int(a1);

    //cerr << "Channel-Type(" << int(i) << "): " << int(a1) << endl;
    i++;
    a1 = (unsigned char)fgetc(dz);                // just to jump over 'reverred to which bytes'
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    a3 = (unsigned char)fgetc(dz);
    a4 = (unsigned char)fgetc(dz);

    //cerr << "a1: " << int(a1) << " - a2: " << int(a2) << " - a3: " << int(a3) << " - a4: " << int(a4) << endl;
    jump_width = octals2int(0, 0, a1, a2) + 1;
    if (a3 > 0)
      jump_width *= 2;

    //cerr << "Jump_Width: " << int( jump_width ) << endl << endl;
    fseek(dz, int(jump_width), SEEK_CUR);         // jumping over channel data
  }

  number_of_rgba_channels = 0;
  if (number_of_channels > 3)
  {
    for (j = 0; (j < number_of_channels - 3); j++)
    {
      if (channel_types[j] == 1 && channel_types[j + 1] == 2 && channel_types[j + 2] == 3 && channel_types[j + 3] == 4)
      {
        number_of_rgba_channels++;
      }
    }
  }

  // cerr << "number of rgba channels: " << int(number_of_rgba_channels) << endl;
  number_of_tf_channels = 0;
  if (number_of_channels > 0)
  {
    for (j = 0; j < number_of_channels; j++)
    {
      if (channel_types[j] > 4 && channel_types[j] < 9)
      {
        number_of_tf_channels++;
      }
    }
  }

  // cerr << "number of tf channels: " << int(number_of_tf_channels) << endl;
  delete[] channel_types;

  number_of_pin_lists = int(fgetc(dz));
  if (vvDebugMsg::isActive(3))
    cerr << "readFile: number of pin lists: " << int(number_of_pin_lists) << "					[ OK ]" << endl;
  number_of_pins = new int[number_of_pin_lists];
  for (j = 0; j < number_of_pin_lists; j++)
  {
    a1 = (unsigned char)fgetc(dz);
    if (vvDebugMsg::isActive(3))
    {
      cerr << "readFile: accessing pin list " << int(j + 1) << "(" << int(a1) << ") out of " << int(number_of_pin_lists) << " pin lists			[ OK ]" << endl;
    }

    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    number_of_pins[j] = octals2int(0, 0, a1, a2);
    if (vvDebugMsg::isActive(3))
      cerr << "readFile: number of pins: " << int(number_of_pins[j]) << "						[ OK ]" << endl;

    if (vvDebugMsg::isActive(3))
      cerr << "readFile: jumping over pins ...";
    a1 = (unsigned char)fgetc(dz);
                                                  // description text
    fseek(dz, octals2int(0, 0, 0, a1), SEEK_CUR);
                                                  // pin-entries
    fseek(dz, int(17 * number_of_pins[j]), SEEK_CUR);
    if (vvDebugMsg::isActive(3))
      cerr << "						[ OK ]" << endl;
  }

  // checksum
  a1 = (unsigned char)fgetc(dz);

  // SUCHMARKE
  if (int(a1) == checksum)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "readFile: Checksum-Test: " << int(a1) << " 						[ OK ]" << endl;
  }
  else
  {
    cerr << "## Error: Checksum incorrect. Is " << int(a1) << " but should be " << checksum << "			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  // EOF
  a1 = (unsigned char)fgetc(dz);
  if (a1 == 123)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "readFile: End Of File reached: 123					[ OK ]" << endl;
  }
  else
  {
    cerr << "## Error: End of File wrong. Is " << int(a1) << " but should be 123			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  FILEREAD = 1;
  fclose(dz);
  return errnmb;
}

int vvvffile::writeFile(void)
{
  unsigned char*  iopuffer;
  errnmb = 0;

  // Check if file exists and ask for overwrite-confirmation
  // SUCHMARKE
  // Open file for writing
  dz = fopen(fileName, "wb");
  if (!dz)
  {
    cerr << "## Error: Unable to open file for writing		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  // Write identity
  fputc(86, dz);                                  // V
  fputc(70, dz);                                  // F
  fputc(45, dz);                                  // -
  fputc(70, dz);                                  // F
  fputc(105, dz);                                 // i
  fputc(108, dz);                                 // l
  fputc(101, dz);                                 // e
  fputc(32, dz);                                  //
  fputc(118, dz);                                 // v
  fputc((int(version_number / 100) + 48), dz);    // x
  fputc(46, dz);                                  // .
                                                  // x
  fputc((int(version_number / 10) % 10 + 48), dz);
  fputc((version_number % 10 + 48), dz);          // x

  if (vvDebugMsg::isActive(3))
  {
    cerr <<
      "writeFile: Identity written: VF-File v" <<
      int(version_number / 100) <<
      "." <<
      int(version_number / 10) %
      10 <<
      version_number %
      10 <<
      "				[ OK ]" <<
      endl;
  }

  checksum = 0;
  proceeded = 13;

  // Write headersize, make no use of unassigned area
  a1 = (unsigned char)((headersize & (255 << 8)) >> 8);
  a2 = (unsigned char)(headersize & (255));
  fputc(a1, dz);
  fputc(a2, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Headersize written: " << headersize << "					[ OK ]" << endl;

  checksum += headersize;
  proceeded += 2;

  // Set Number of Data-Arrays
  if (temp_array_number < 1)
  {
    cerr << "## Error: number of arrays has stupid value: " << temp_array_number << "		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  number_of_data_arrays = temp_array_number;
  int2octals(&a1, &a2, &a3, &a4, number_of_data_arrays);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Number of data arrays: " << number_of_data_arrays << "					[ OK ]" << endl;

  checksum += a1 + a2 + a3 + a4;
  proceeded = 2;

  // write volume dimensions
  int2octals(&a1, &a2, &a3, &a4, voxel_number_x);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Volume dimensions X: " << voxel_number_x << "					[ OK ]" << endl;

  checksum += (a1 + a2 + a3 + a4);
  proceeded += 4;

  int2octals(&a1, &a2, &a3, &a4, voxel_number_y);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Volume dimensions Y: " << voxel_number_y << "					[ OK ]" << endl;

  checksum += (a1 + a2 + a3 + a4);
  proceeded += 4;

  int2octals(&a1, &a2, &a3, &a4, voxel_number_z);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Volume dimensions Z: " << voxel_number_z << "					[ OK ]" << endl;

  checksum += (a1 + a2 + a3 + a4);
  proceeded += 4;

  // write number of bytes every voxel is described with
  fputc(bytes_per_voxel, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Bytes per Voxel: " << bytes_per_voxel << "						[ OK ]" << endl;

  checksum += bytes_per_voxel;
  proceeded += 1;

  // Place for possible extensions, currently unassigned
  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Writing extensions: none					[ OK ]" << endl;

  checksum += 0;
  proceeded += 0;

  // writing checksum
  checksum = checksum % 256;
  fputc(checksum, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Writing Checksum: " << checksum << "					[ OK ]" << endl;

  // set type of icon
  fputc(icontype, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Set type of icon: " << icontype << "						[ OK ]" << endl;

  // write icon
  switch (icontype)
  {
    case 0:
      if (vvDebugMsg::isActive(3))
        cerr << "writeFile: no icon set, no data written:				[ OK ]" << endl;
      break;
    case 1:
      for (i = 0; i < 3072; i++)
        fputc(icondata[i], dz);
      if (vvDebugMsg::isActive(3))
        cerr << "writeFile: icon: 32x32, 24bit, rgb, bitmap				[ OK ]" << endl;
      break;
    case 2:
      for (i = 0; i < 64 * 64 * 3; i++)
        fputc(icondata[i], dz);
      if (vvDebugMsg::isActive(3))
        cerr << "writeFile: icon: 64x64, 24bit, rgb, bitmap				[ OK ]" << endl;
      break;
    default:
      if (vvDebugMsg::isActive(3))
      {
        cerr << "## Error: only icontype 0 or 1 supported		[ FAILED ]" << endl;
        errnmb = 1;
        return errnmb;
      }

      //         break; // unreachable code
  }

  // set text length
  a1 = (unsigned char)((textlength & (255 << 8)) >> 8);
  a2 = (unsigned char)(textlength & (255));
  fputc(a1, dz);
  fputc(a2, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Set length of text: " << textlength << "					[ OK ]" << endl;

  // write text
  for (i = 0; i < textlength; i++)
    fputc(descr[i], dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Writing Text: " << int(strlen(descr)) << " characters					[ OK ]" << endl;

  // set checkbit #1
  fputc(66, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Set first checkbit: 						[ OK ]" << endl;

  // write data
  if (!tempfile)
  {
    cerr << "## Error: no tempfile specified						[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: opening tempfile: " << tempfile << "			[ OK ]" << endl;

  // open tempfile
  dt = fopen(tempfile, "rb");
  if (!dt)
  {
    cerr << "## Error: unable to open tempfile: " << tempfile << "		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  // copy tempfile to 'real' file
  iopuffer = new unsigned char[4096];
  for (i = 1; i < (temp_array_number + 1); i++)
  {
    a1 = (unsigned char)fgetc(dt);
    a2 = (unsigned char)fgetc(dt);
    a3 = (unsigned char)fgetc(dt);
    a4 = (unsigned char)fgetc(dt);
    k = octals2int(a1, a2, a3, a4) + 5;
    int2octals(&a1, &a2, &a3, &a4, k);
    fputc(a1, dz);
    fputc(a2, dz);
    fputc(a3, dz);
    fputc(a4, dz);

    if (vvDebugMsg::isActive(3))
      cerr << "writeFile: copy data array " << i << " of " << temp_array_number << " (" << int(k) << " bytes)";
    j = 0;
    int retval;
    while ((k - j) > 4096)
    {
      retval=fread(iopuffer, 1, 4096, dt);
      if (retval!=4096)
      {
        std::cerr<<"vvvffile::writeFile: fread failed"<<std::endl;
        delete[] iopuffer;
        return 1;
      }
      retval=fwrite(iopuffer, 1, 4096, dz);
      if (retval!=4096)
      {
        std::cerr<<"vvvffile::writeFile: fwrite failed"<<std::endl;
        delete[] iopuffer;
        return 1;
      }
      j += 4096;
    }

    retval=fread(iopuffer, 1, (k - j), dt);
    if (retval!=k-j)
    {
      std::cerr<<"vvvffile::writeFile: fread failed"<<std::endl;
      delete[] iopuffer;
      return 1;
    }
    retval=fwrite(iopuffer, 1, (k - j), dz);
    if (retval!=k-j)
    {
      std::cerr<<"vvvffile::writeFile: fwrite failed"<<std::endl;
      delete[] iopuffer;
      return 1;
    }
    if (vvDebugMsg::isActive(3))
      cerr << "			[ OK ]" << endl;
  }

  delete[] iopuffer;

  // close tempfile
  fclose(dt);
  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: closing tempfile " << tempfile << "			[ OK ]" << endl;

  // delete tempfile
  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: deleting tempfile 						[ OK ]" << endl;

  // set checkbit #2
  fputc(99, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Set second checkbit:						[ OK ]" << endl;

  // set tail-size
  tailsize = 28 + channel_array_length;           // with unassigned_space=0
  int2octals(&a1, &a2, &a3, &a4, tailsize);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Set tail-size: " << tailsize << "						[ OK ]" << endl;

  // set x-coordinate of hotspot
  int2octals(&a1, &a2, &a3, &a4, hotspot_x);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Write x-coordinate of hotspot: " << hotspot_x << "				[ OK ]" << endl;

  // set y-coordinate of hotspot
  int2octals(&a1, &a2, &a3, &a4, hotspot_y);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Write y-coordinate of hotspot: " << hotspot_y << "				[ OK ]" << endl;

  // set z-coordinate of hotspot
  int2octals(&a1, &a2, &a3, &a4, hotspot_z);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Write z-coordinate of hotspot: " << hotspot_z << "				[ OK ]" << endl;

  // set realsize of voxels in x-direction
  a1 = a2 = a3 = a4 = 0;
  float2octals(&a1, &a2, &a3, &a4, realsize_x);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Saving real dimensions (x): " << realsize_x << "				[ OK ]" << endl;

  // set realsize of voxels in y-direction
  a1 = a2 = a3 = a4 = 0;
  float2octals(&a1, &a2, &a3, &a4, realsize_y);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Saving real dimensions (y): " << realsize_y << "				[ OK ]" << endl;

  // set realsize of voxels in z-direction
  a1 = a2 = a3 = a4 = 0;
  float2octals(&a1, &a2, &a3, &a4, realsize_z);
  fputc(a1, dz);
  fputc(a2, dz);
  fputc(a3, dz);
  fputc(a4, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Saving real dimensions (z): " << realsize_z << "				[ OK ]" << endl;

  // write channel
  if (channel_array_length == 1)
  {
    fputc(0, dz);

    if (vvDebugMsg::isActive(3))
      cerr << "writeFile: Write Channels: none						[ OK ]" << endl;
  }
  else
  {
    for (i = 0; i < channel_array_length; i++)
      fputc(channel[i], dz);

    if (vvDebugMsg::isActive(3))
      cerr << "writeFile: Write Channels: " << int(channel[0]) << "						[ OK ]" << endl;
  }

  // TEST
  if (number_of_pin_lists == 0)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "writeFile: no pin lists to be saved						[ OK ]" << endl;
    fputc(0, dz);
  }
  else
  {
    if (vvDebugMsg::isActive(3))
      cerr << "writeFile: saving pin lists, " << int(pin_data_length) << " bytes 					[ OK ]" << endl;
    int retval;
    retval=fwrite(pin_data, 1, pin_data_length, dz);
    if (retval!=pin_data_length)
    {
      std::cerr<<"vvvffile::writeFile: fwrite failed"<<std::endl;
      return 1;
    }

    /*for( j=0; j<number_of_pin_lists; j++ )
    {
    int2octals( &a1, &a2, &a3, &a4, number_of_pins[j] );
    if( a1*a2!=0 )
      cerr << "## Error: writeFile: number of pins exceeding 65535				[ FAILED ]" << endl;
    fputc( a3, dz );
    fputc( a4, dz );
    if (vvDebugMsg::isActive(3))
      cerr << "writeFile: Write pins: " << int( number_of_pins[j] ) << " ...";
    fwrite( pin_data, 1, pin_data_length, dz );
    if (vvDebugMsg::isActive(3))
    cerr << " 					[ OK ]" << endl;
    }*/
  }

  // Place to put further extensions, none yet
  // Write checksum 2
  checksum = checksum % 256;
  fputc(checksum, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Writing Checksum: " << checksum << "					[ OK ]" << endl;

  // Set EOF
  fputc(123, dz);

  if (vvDebugMsg::isActive(3))
    cerr << "writeFile: Writing End Of File:						[ OK ]" << endl;

  // Close file and return
  fclose(dz);
  return errnmb;
}

int vvvffile::getNumberOfDataArrays(void)
{
  if (!FILEREAD)
    readFile();

  return number_of_data_arrays;
}

int vvvffile::getNumberOfRGBAChannels(void)
{
  if (!FILEREAD)
    readFile();

  return number_of_channels;
}

int vvvffile::setArrayDimensions(int x, int y, int z, int b)
{
  errnmb = 0;
  voxel_number_x = x;
  voxel_number_y = y;
  voxel_number_z = z;
  bytes_per_voxel = b;
  return errnmb;
}

int vvvffile::getArrayDimensions(int* x, int* y, int* z, int* b)
{
  errnmb = 0;
  if (!FILEREAD)
    readFile();

  *x = voxel_number_x;
  *y = voxel_number_y;
  *z = voxel_number_z;
  *b = bytes_per_voxel;
  return errnmb;
}

int vvvffile::setHotspot(int hotx, int hoty, int hotz)
{
  errnmb = 0;
  hotspot_x = hotx;
  hotspot_y = hoty;
  hotspot_z = hotz;
  return errnmb;
}

int vvvffile::getHotspot(int* hotx, int* hoty, int* hotz)
{
  errnmb = 0;
  if (!FILEREAD)
    readFile();

  *hotx = hotspot_x;
  *hoty = hotspot_y;
  *hotz = hotspot_z;
  return errnmb;
}

int vvvffile::setVoxelsize(float voxx, float voxy, float voxz)
{
  errnmb = 0;
  realsize_x = voxx;
  realsize_y = voxy;
  realsize_z = voxz;
  return errnmb;
}

int vvvffile::getVoxelsize(float* voxx, float* voxy, float* voxz)
{
  errnmb = 0;
  if (!FILEREAD)
    readFile();

  *voxx = realsize_x;
  *voxy = realsize_y;
  *voxz = realsize_z;
  return errnmb;
}

int vvvffile::setFileDescription(char* text)
{
  errnmb = 0;
  if (descr)
    delete[] descr;
  descr = new char[strlen(text) + 1];
  strcpy(descr, text);
  textlength = strlen(descr);
  return errnmb;
}

char *vvvffile::getFileDescription(void)
{
  if (descr)
    delete[] descr;
  if (!FILEREAD)
    readFile();

  dz = fopen(fileName, "rb");
  if (!dz)
    cerr << "## Error: Unable to open file for reading		[ FAILED ]" << endl;
  else if (vvDebugMsg::isActive(3))
    cerr << "getFileDescription: open file for LOADing: " << fileName << "[ OK ]" << endl;

  if (vvDebugMsg::isActive(3))
    cerr << "getFileDescription: reading text (" << textlength << " bytes)";
  fseek(dz, int(headersize + 2), SEEK_CUR);       // header+checksum
  fseek(dz, int(iconsize), SEEK_CUR);             // icondata
  fseek(dz, 2L, SEEK_CUR);                        // text-length

  descr = new char[textlength + 1];
  for (i = 0; i < textlength; i++)
    descr[i] = char(fgetc(dz));

  fclose(dz);
  if (vvDebugMsg::isActive(3))
    cerr << "					[ OK ]" << endl;
  return descr;
}

int vvvffile::setIcon(int type, unsigned char* data)
{
  errnmb = 0;
  switch (type)
  {
    case 0:
      icontype = type;
      icondata = NULL;
      break;
    case 1:
      icontype = type;
      icondata = new unsigned char[3072];
      for (i = 0; i < 3072; i++)
        icondata[i] = data[i];
      break;
    case 2:
      icontype = type;
      icondata = new unsigned char[64 * 64 * 3];
      for (i = 0; i < 64 * 64 * 3; i++)
        icondata[i] = data[i];
      break;
    default:
      icontype = 0;
      icondata = NULL;
      cerr << "## Error: setIcon: unsupported icon type " << type << endl;
      errnmb = 1;
      return errnmb;

      //          break; // unreachable code
  }

  return errnmb;
}

int vvvffile::getIconSize(void)
{
  int icondatalength;
  if (!FILEREAD)
    readFile();
  switch (icontype)
  {
    case 0:   icondatalength = 0; if (vvDebugMsg::isActive(3)) cerr << "getIconSize: no icon, size: 0						[ OK ]" << endl; break;
    case 1:   icondatalength = 3072; if (vvDebugMsg::isActive(3)) cerr << "getIconSize: 32x32 rgb, size: 3072					[ OK ]" << endl; break;
    case 2:   icondatalength = 64 * 64 * 3; if (vvDebugMsg::isActive(3)) cerr << "getIconSize: 64x64 rgb					[ OK ]" << endl; break;
    default:  icondatalength = 0; cerr << "## Error: getIcon: unsupported icon type " << icontype << endl; break;
  }

  return icondatalength;
}

int vvvffile::getIcon(unsigned char* icondata)
{
  errnmb = 0;
  if (!FILEREAD)
    readFile();
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "getIcon: open file for LOADing: " << fileName << "			[ OK ]" << endl;

  fseek(dz, int(headersize + 1), SEEK_CUR);       // header+checksum
  a1 = (unsigned char)fgetc(dz);
  if (a1 == icontype)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "getIcon: begin of icon found, icontype: " << int(a1) << "				[ OK ]" << endl;
  }

  if (a1 == 0)
  {
    cerr << "# Warning: no icon included, do not try to get its data			[ WRN ]" << endl;
  }
  else
  {
    int retval;
    retval=fread(icondata, 1, iconsize, dz);
    if (retval!=iconsize)
    {
      std::cerr<<"vvvffile::writeFile: fread failed"<<std::endl;
      return 1;
    }
  }

  fclose(dz);
  return errnmb;
}

/*
unsigned char *vvvffile::getIcon( int *type, int *iconlength )
{
   *type=icontype;
   switch( icontype )
   {
      case 0:
        *iconlength=0;
       break;
      case 1:
        *iconlength=3072;
break;
default:
*iconlength=0;
cerr << "## Error: getIcon: unsupported icon type " << icontype << endl;
break;
}
return( icondata );
}
*/
int vvvffile::setDataFile(void)
{
  errnmb = 0;
  temp_array_number = 0;

  // resolve name of tempfile
  if (tempfile == NULL)
  {
    tempfile = new char[strlen(fileName) + 2];
    strcpy(tempfile, fileName);
    for (i = strlen(tempfile); tempfile[i] != '.' && i; i--)
      ;
    tempfile[i + 1] = 't';
    tempfile[i + 2] = 'm';
    tempfile[i + 3] = 'p';
    tempfile[i + 4] = 0;
    if (vvDebugMsg::isActive(3))
      cerr << "SetDataFile: name of tempfile: " << tempfile << "			[ OK ]" << endl;
  }

  // Open file for writing
  dt = fopen(tempfile, "wb");
  if (!dt)
  {
    cerr << "## Error: Unable to open file for writing		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "SetDataFile: creating tempfile						[ OK ]" << endl;

  fclose(dt);
  return errnmb;
}

int vvvffile::addToDataFile(int data_array_length, unsigned char coding, unsigned char* data)
{
  errnmb = 0;
  if (!tempfile)
  {
    cerr << "## Error: no tempfile specified						[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "addToDataFile: name of tempfile: " << tempfile << "		[ OK ]" << endl;

  // Open file for append-writing
  dt = fopen(tempfile, "ab");
  if (!dt)
    cerr << "## Error: Unable to open file for writing		[ FAILED ]" << endl;
  else if (vvDebugMsg::isActive(3))
    cerr << "addToDataFile: open tempfile						[ OK ]" << endl;

  // write data_array_length
  if (vvDebugMsg::isActive(3))
    cerr << "addToDataFile: writing length of data array: " << data_array_length << "			[ OK ]" << endl;
  int2octals(&a1, &a2, &a3, &a4, data_array_length);
  fputc(a1, dt);
  fputc(a2, dt);
  fputc(a3, dt);
  fputc(a4, dt);

  // write time_stamp
  if (vvDebugMsg::isActive(3))
    cerr << "addToDataFile: writing time stamp: " << time_stamp << "					[ OK ]" << endl;
  float2octals(&a1, &a2, &a3, &a4, (float)time_stamp);
  fputc(a1, dt);
  fputc(a2, dt);
  fputc(a3, dt);
  fputc(a4, dt);

  // write type of coding
  if (vvDebugMsg::isActive(3))
  {
    cerr << "addToDataFile: type of coding: ";
    if (coding == 0)
      cerr << "unencoded				[ OK ]" << endl;
    else
      cerr << "coded, type " << int(coding) << "				[ OK ]" << endl;
  }

  fputc(coding, dt);

  // write data_array
  if (vvDebugMsg::isActive(3))
    cerr << "addToDataFile: writing data: " << int(data_array_length) << " bytes";
  int retval;
  retval=fwrite(data, 1, data_array_length, dt);
  if (retval!=data_array_length)
  {
    std::cerr<<"vvvffile::addToDataFile: fwrite failed"<<std::endl;
    return 1;
  }
  if (vvDebugMsg::isActive(3))
    cerr << " 				[ OK ]" << endl;

  // of course close tempfile after completion
  if (vvDebugMsg::isActive(3))
    cerr << "addToDataFile: closing tempfile						[ OK ]" << endl;
  fclose(dt);
  temp_array_number += 1;
  return errnmb;
}

int vvvffile::delDataFile(void)
{
  errnmb = 0;
  if (vvDebugMsg::isActive(3))
    cerr << "DelDataFile: deleting tempfile: " << tempfile << "			[ OK ]" << endl;
  if (tempfile != NULL)
  {
    unlink(tempfile);
    delete[] tempfile;
    tempfile = NULL;
  }

  temp_array_number = 0;
  return errnmb;
}

int vvvffile::setRGBAChannels(void)
{
  errnmb = 0;
  if (channel)
    delete[] channel;
  number_of_channels = 0;

  //   number_of_rbga_channels=0;
  channel_array_length = 1;
  channel = new unsigned char[channel_array_length];
  channel[0] = (unsigned char)number_of_channels;
  return errnmb;
}

int vvvffile::addRGBAChannel(unsigned char* r_channel, unsigned char* g_channel, unsigned char* b_channel, unsigned char* a_channel, char* description)
{
  int             jumper, old_length, new_length, description_length;
  unsigned char*  tempchannel;
  errnmb = 0;

  // prepare everything needed
  description_length = strlen(description);
  old_length = channel_array_length;
  new_length = old_length + 263 * 4 + description_length;
  jumper = old_length;

  // enlarge channel to place further rgba's in it
  tempchannel = new unsigned char[old_length];
  for (i = 0; i < old_length; i++)
    tempchannel[i] = channel[i];
  delete[] channel;
  channel = new unsigned char[new_length];
  for (i = 0; i < old_length; i++)
    channel[i] = tempchannel[i];
  delete[] tempchannel;

  // increase number of channels by one and save new value
  number_of_channels += 4;
  number_of_rgba_channels++;
  if (number_of_channels > 255)
  {
    cerr << "## Error: trying to save more then 255 RGBA-Channels			[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  channel[0] = (unsigned char)number_of_channels;

  // now append rgba's to channel
  if (description_length > 255)
  {
    description_length = 255;
    cerr << "# Warning: addRGBAChannel: Description Text has more than 255 characters, cutting end...	[ WRN ]" << endl;
  }

  // description length
  channel[jumper++] = (unsigned char)description_length;

  // description text
  for (i = 0; i < description_length; i++)
    channel[jumper + i] = description[i];
  jumper += description_length;

  // type of channel: red=1
  channel[jumper++] = 1;

  // describing which voxel-bytes
  channel[jumper++] = 1;

  // x-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // y-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // data of red channel
  for (i = 0; i < 256; i++)
    channel[jumper + i] = r_channel[i];
  jumper += 256;

  // length of channel description
  channel[jumper++] = 0;

  // type of channel: green=2
  channel[jumper++] = 2;

  // describing which voxel-bytes
  channel[jumper++] = 1;

  // x-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // y-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // data of green channel
  for (i = 0; i < 256; i++)
    channel[jumper + i] = g_channel[i];
  jumper += 256;

  // length of channel description
  channel[jumper++] = 0;

  // type of channel: blue=3
  channel[jumper++] = 3;

  // describing which voxel-bytes
  channel[jumper++] = 1;

  // x-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // y-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // data of blue channel
  for (i = 0; i < 256; i++)
    channel[jumper + i] = b_channel[i];
  jumper += 256;

  // length of channel description
  channel[jumper++] = 0;

  // type of channel: alpha=4
  channel[jumper++] = 4;

  // describing which voxel-bytes
  channel[jumper++] = 1;

  // x-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // y-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // data of alpha channel
  for (i = 0; i < 256; i++)
    channel[jumper + i] = a_channel[i];
  jumper += 256;

  channel_array_length = new_length;

  return errnmb;
}

int vvvffile::getRGBAChannel(int number, unsigned char*  r_channel, unsigned char*  g_channel, unsigned char*  b_channel, unsigned char*  a_channel,
char*  channel_description)
{
  errnmb = 0;
  if (!FILEREAD)
    readFile();
  if (number > number_of_rgba_channels)
  {
    cerr << "## Error: trying to get rgba-channel " << number << " out of just " << number_of_rgba_channels << endl;
    errnmb = 1;
    return errnmb;
  }

  if (number_of_rgba_channels < 1)
    cerr << "# Warning: File has no rgba-channels, do not try to get one			[ WRN ]" << endl;
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "getRGBAChannel: open file for LOADing: " << fileName << "		[ OK ]" << endl;

  fseek(dz, int(headersize + 2), SEEK_CUR);       // header+checksum
  fseek(dz, int(iconsize), SEEK_CUR);             // icondata
  fseek(dz, int(2 + textlength), SEEK_CUR);       // text-length+text+checkbit
  if (fgetc(dz) == 66)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "getRGBAChannel: Checkbit 1 found					[ OK ]" << endl;
    else
    {
      cerr << "## Error: getRGBAChannel: Checkbit 1 not correct		[ FAILED ]" << endl;
      errnmb = 1;
      return errnmb;
    }
  }

  if (vvDebugMsg::isActive(3))
    cerr << "getRGBAChannel: jumping over data ..";
  for (i = 0; i < number_of_data_arrays; i++)
  {
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    a3 = (unsigned char)fgetc(dz);
    a4 = (unsigned char)fgetc(dz);
    fseek(dz, int(octals2int(a1, a2, a3, a4)), SEEK_CUR);
  }

  if (vvDebugMsg::isActive(3))
    cerr << "					[ OK ]" << endl;
  if (fgetc(dz) == 99)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "getRGBAChannel: Checkbit 2 found					[ OK ]" << endl;
    else
    {
      cerr << "## Error: getRGBAChannel: Checkbit 2 not correct		[ FAILED ]" << endl;
      errnmb = 1;
      return errnmb;
    }
  }

  fseek(dz, 29L, SEEK_CUR);                       // tail-entries to first channel

  for (i = 1; i < number; i++)
  {
    a1 = (unsigned char)fgetc(dz);
    fseek(dz, int(a1 - 1), SEEK_CUR);             // description text
    fseek(dz, int(263 * 4), SEEK_CUR);            // rgba-data
  }

  a1 = (unsigned char)fgetc(dz);
  if (channel_description)
    delete[] channel_description;
  channel_description = new char[int(a1) + 1];

  // SUCHMARKE
  if (channel_description)
    delete[] channel_description;

  size_t retval=0;
  fseek(dz, int(a1 - 1), SEEK_CUR);
  fseek(dz, 7L, SEEK_CUR);
  retval+=fread(r_channel, 1, 256, dz);
  fseek(dz, 7L, SEEK_CUR);
  retval+=fread(g_channel, 1, 256, dz);
  fseek(dz, 7L, SEEK_CUR);
  retval+=fread(b_channel, 1, 256, dz);
  fseek(dz, 7L, SEEK_CUR);
  retval+=fread(a_channel, 1, 256, dz);

  if (retval!=1024)
  {
    std::cerr<<"vvvffile::getRGBAChannel: fread failed"<<std::endl;
    return 1;
  }

  fclose(dz);
  return errnmb;
}

int vvvffile::delRGBAChannels(void)
{
  errnmb = 0;
  if (channel)
    delete[] channel;
  number_of_channels = 0;
  number_of_rgba_channels = 0;
  channel_array_length = 1;
  return errnmb;
}

int vvvffile::initPins(void)
{
  errnmb = 0;
  if (pin_data)
    delete[] pin_data;
  pin_data_length = 1;
  number_of_pin_lists = 0;
  pin_data = new unsigned char[pin_data_length];
  pin_data[0] = 0;
  return errnmb;
}

// add a pin to array
int vvvffile::addPin(int list_number, int type, float value1, float value2, float value3, float xPos, char* description)
{
  unsigned char*  temp_array;
  int old_length, marker = 0, marker2 = 0, descr_len;
  errnmb = 0;
  old_length = pin_data_length;
  temp_array = new unsigned char[old_length + 1];
  for (i = 0; i < old_length; i++)
    temp_array[i] = pin_data[i];
  delete[] pin_data;

  if (!description)
    descr_len = 0;
  else
    descr_len = int(strlen(description));
  if (descr_len > 255)
  {
    cerr << "# Warning: addPin: description text must not have more than 255 character, skipping rest		[ WRN ]" << endl;
    descr_len = 255;
  }

  if (list_number > (number_of_pin_lists + 1))
  {
    cerr <<
      "# Warning: addPin: new pin list added as number: " <<
      int(number_of_pin_lists + 1) <<
      " instead of wanted " <<
      int(list_number) <<
      "	[ WRN ]" <<
      endl;
    list_number = number_of_pin_lists + 1;
  }

  if (number_of_pin_lists == 0)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "addPin: adding first pin, first pin list				[ OK ]" << endl;
    delete[] temp_array;
    pin_data_length = (22 + descr_len);
    pin_data = new unsigned char[pin_data_length];
    number_of_pin_lists = 1;
    pin_data[0] = (unsigned char)number_of_pin_lists;
    pin_data[1] = 1;                              // number of this pin list
    pin_data[2] = 0;
    pin_data[3] = 1;                              // number of pins in it
    pin_data[4] = (unsigned char)(descr_len % 255);
    for (i = 0; i < descr_len; i++)
      pin_data[5 + i] = description[i];
    marker = 5 + descr_len;
    pin_data[marker] = (unsigned char)type;

    float2octals(&a1, &a2, &a3, &a4, value1);
    pin_data[marker + 1] = a1;
    pin_data[marker + 2] = a2;
    pin_data[marker + 3] = a3;
    pin_data[marker + 4] = a4;
    float2octals(&a1, &a2, &a3, &a4, value2);
    pin_data[marker + 5] = a1;
    pin_data[marker + 6] = a2;
    pin_data[marker + 7] = a3;
    pin_data[marker + 8] = a4;
    float2octals(&a1, &a2, &a3, &a4, value3);
    pin_data[marker + 9] = a1;
    pin_data[marker + 10] = a2;
    pin_data[marker + 11] = a3;
    pin_data[marker + 12] = a4;
    float2octals(&a1, &a2, &a3, &a4, xPos);
    pin_data[marker + 13] = a1;
    pin_data[marker + 14] = a2;
    pin_data[marker + 15] = a3;
    pin_data[marker + 16] = a4;
  }
  else
  {
    if (list_number < (number_of_pin_lists + 1))
    {
      int jump_len;

      if (descr_len != 0)
        cerr << "# Warning: addPin: list " << int(list_number) << " exists, ignoring description			[ WRN ]" << endl;
      marker = 0;
      pin_data_length += 17;
      pin_data = new unsigned char[pin_data_length];
      marker++;                                   // number of pin lists
      for (i = 0; i < (list_number - 1); i++)
      {
        marker++;                                 // list number
        jump_len = (temp_array[marker] * 256 + temp_array[marker + 1]) * 17;
        marker += 2;
        marker += (temp_array[marker] + 1);       // description text
        marker += jump_len;
      }

      if (temp_array[marker] == list_number)
      {
        if (vvDebugMsg::isActive(3))
          cerr << "addPin: found correct pin list: " << int(list_number) << "					[ OK ]" << endl;
      }
      else
      {
        cerr <<
          "## Error: addPin: couldn't find correct pin list. should be " <<
          int(list_number) <<
          " but is " <<
          int(pin_data[marker]) <<
          "		[ FAILED ]" <<
          endl;
        errnmb = 1;
        return errnmb;
      }

      marker++;                                   // list number

      for (i = 0; i < marker; i++)
        pin_data[i] = temp_array[i];
      jump_len = (temp_array[marker] * 256 + temp_array[marker + 1]);
      int2octals(&a1, &a2, &a3, &a4, (jump_len + 1));
      jump_len *= 17;
      pin_data[marker] = a3;
      pin_data[marker + 1] = a4;
      marker += 2;
      marker2 = marker;
      marker += (temp_array[marker] + 1);         // description text
      marker += jump_len;
      for (i = marker2; i < marker; i++)
        pin_data[i] = temp_array[i];
      marker2 = marker;
      pin_data[marker] = (unsigned char)type;
      float2octals(&a1, &a2, &a3, &a4, value1);
      pin_data[marker + 1] = a1;
      pin_data[marker + 2] = a2;
      pin_data[marker + 3] = a3;
      pin_data[marker + 4] = a4;
      float2octals(&a1, &a2, &a3, &a4, value2);
      pin_data[marker + 5] = a1;
      pin_data[marker + 6] = a2;
      pin_data[marker + 7] = a3;
      pin_data[marker + 8] = a4;
      float2octals(&a1, &a2, &a3, &a4, value3);
      pin_data[marker + 9] = a1;
      pin_data[marker + 10] = a2;
      pin_data[marker + 11] = a3;
      pin_data[marker + 12] = a4;
      float2octals(&a1, &a2, &a3, &a4, xPos);
      pin_data[marker + 13] = a1;
      pin_data[marker + 14] = a2;
      pin_data[marker + 15] = a3;
      pin_data[marker + 16] = a4;
      marker += 17;

      for (i = marker2; i < old_length; i++)
        pin_data[i - marker2 + marker] = temp_array[i];

      delete[] temp_array;
    }
    else
    {
      if (vvDebugMsg::isActive(3))
        cerr << "addPin: creating new pin list: " << int(list_number) << "					[ OK ]" << endl;

      pin_data_length += (21 + descr_len);
      pin_data = new unsigned char[pin_data_length];
      for (i = 0; i < old_length; i++)
        pin_data[i] = temp_array[i];
      delete[] temp_array;

      number_of_pin_lists++;
      if (number_of_pin_lists > 255)
      {
        cerr << "## Error: Number of saved pin lists must be < 256			[ FAILED ]" << endl;
        errnmb = 1;
        return errnmb;
      }

      pin_data[0] = (unsigned char)(number_of_pin_lists % 256);
      marker = old_length;
      pin_data[marker] = (unsigned char)list_number;
      pin_data[marker + 1] = 0;
      pin_data[marker + 2] = 1;
      marker += 3;
      if (descr_len == 0)
      {
        pin_data[marker] = 0;
        marker++;
      }
      else
      {
        pin_data[marker] = (unsigned char)(descr_len);
        for (i = 0; i < descr_len; i++)
          pin_data[marker + 1 + i] = description[i];
        marker += (1 + descr_len);
      }

      pin_data[marker] = (unsigned char)type;
      float2octals(&a1, &a2, &a3, &a4, value1);
      pin_data[marker + 1] = a1;
      pin_data[marker + 2] = a2;
      pin_data[marker + 3] = a3;
      pin_data[marker + 4] = a4;
      float2octals(&a1, &a2, &a3, &a4, value2);
      pin_data[marker + 5] = a1;
      pin_data[marker + 6] = a2;
      pin_data[marker + 7] = a3;
      pin_data[marker + 8] = a4;
      float2octals(&a1, &a2, &a3, &a4, value3);
      pin_data[marker + 9] = a1;
      pin_data[marker + 10] = a2;
      pin_data[marker + 11] = a3;
      pin_data[marker + 12] = a4;
      float2octals(&a1, &a2, &a3, &a4, xPos);
      pin_data[marker + 13] = a1;
      pin_data[marker + 14] = a2;
      pin_data[marker + 15] = a3;
      pin_data[marker + 16] = a4;
      marker += 17;
    }
  }

  /*
  unsigned char *temp_array;
  int old_length=pin_data_length;
  int descr_len;
  if( !description )
   descr_len=0;
  else
   descr_len=int(strlen(description));
  temp_array = new unsigned char[pin_data_length];
  for( i=0; i<pin_data_length; i++ )
   temp_array[i]=pin_data[i];
  delete[] pin_data;

  if( list_number>number_of_pin_lists )
  {
  if( list_number!=number_of_pin_lists+1 )
  {
  cerr << "# Warning: addPin: adding new pin list using number " << int( number_of_pin_lists+1) << " instead of wanted " << int(list_number) << "			[ WRN ]" << endl;
  list_number=number_of_pin_lists+1;
  }
  number_of_pin_lists++;
  if( number_of_pin_lists>255 )
  cerr << "## Error: addPin: just 255 pin lists supported, do not try to add further ones				[ FAILED ]" << endl;

  //add new pin list
  pin_data_length+=(21+descr_len);
  pin_data = new unsigned char[ pin_data_length ];
  for( i=0; i<old_length; i++ )
  pin_data[i] = temp_array[i];
  delete[] temp_array;
  pin_data[0]=(unsigned char)( number_of_pin_lists );
  pin_data[old_length]=(unsigned char)(list_number);
  pin_data[old_length+1]=0;
  pin_data[old_length+2]=1;
  if( descr_len>255 )
  {
  cerr << "# Warning: addPin: length of description text has more than 255 characters, skipping rest		[ WRN ]" << endl;
  descr_len=255;
  }
  int2octals( &a1, &a2, &a3, &a4, descr_len );
  pin_data[old_length+3]=a3;
  pin_data[old_length+4]=a4;
  for( i=0; i<descr_len; i++ )
  pin_data[old_length+5+i]=description[i];
  int marker=old_length+5+descr_len;
  pin_data[marker+1]=(unsigned char)(type);
  for( i=0; i<marker+1; i++)
  cerr << int(i) << ": " << int(pin_data[i]) << endl;
  float2octals( &a1, &a2, &a3, &a4, value1 );
  pin_data[marker+ 2]=a1; pin_data[marker+ 3]=a2; pin_data[marker+ 4]=a3; pin_data[marker+ 5]=a4;
  float2octals( &a1, &a2, &a3, &a4, value2 );
  pin_data[marker+ 6]=a1; pin_data[marker+ 7]=a2; pin_data[marker+ 8]=a3; pin_data[marker+ 9]=a4;
  float2octals( &a1, &a2, &a3, &a4, value3 );
  pin_data[marker+10]=a1; pin_data[marker+11]=a2; pin_data[marker+12]=a3; pin_data[marker+13]=a4;
  float2octals( &a1, &a2, &a3, &a4, xPos );
  pin_data[marker+14]=a1; pin_data[marker+15]=a2; pin_data[marker+16]=a3; pin_data[marker+17]=a4;

  }
  else
  {
  if (vvDebugMsg::isActive(3))
  cerr << "addPin: jumping to correct pin list ...";
  int marker=1;
  int marker2=0;
  for( i=0; i<list_number; i++ )
  {
  marker+=1; // list number

  marker2=octals2int( 0, 0, temp_array[marker], temp_array[marker+1] ) * 17;
  marker+=2;

  marker2 += temp_array[marker];
  marker+=1;

  marker+=1; // pin type

  marker+=marker2; // jump over pins*17 and text
  }
  if( temp_array[marker]==list_number )
  if (vvDebugMsg::isActive(3))
  cerr << "addPin: successfully found list number: " << int(list_number) << ", adding pin			[ OK ]" << endl;
  else
  cerr << "## Error: could not find list number				[ FAILED ]" << endl;

  pin_data_length+=17;
  pin_data = new unsigned char[pin_data_length];

  for( i=0; i<marker; i++ )
  pin_data[i]=temp_array[i];

  marker2=octals2int( 0, 0, temp_array[marker+1], temp_array[marker+2] );
  marker2++;
  int2octals( &a1, &a2, &pin_data[marker+1], &pin_data[marker+2], marker2 );
  marker+=2;

  marker2=temp_array[marker];
  for( i=0; i<(marker2+1); i++ )
  pin_data[marker+i]=temp_array[marker+i];
  marker+=marker2+1;

  pin_data[marker]=type;

  marker++;
  float2octals( &a1, &a2, &a3, &a4, value1 );
  pin_data[marker+ 2]=a1; pin_data[marker+ 3]=a2; pin_data[marker+ 4]=a3; pin_data[marker+ 5]=a4;
  float2octals( &a1, &a2, &a3, &a4, value2 );
  pin_data[marker+ 6]=a1; pin_data[marker+ 7]=a2; pin_data[marker+ 8]=a3; pin_data[marker+ 9]=a4;
  float2octals( &a1, &a2, &a3, &a4, value3 );
  pin_data[marker+10]=a1; pin_data[marker+11]=a2; pin_data[marker+12]=a3; pin_data[marker+13]=a4;
  float2octals( &a1, &a2, &a3, &a4, xPos );
  pin_data[marker+14]=a1; pin_data[marker+15]=a2; pin_data[marker+16]=a3; pin_data[marker+17]=a4;
  marker--;
  for( i=marker; i<old_length; i++ )
  pin_data[i+17]=temp_array[i];
  delete[] temp_array;
  }*/
  /*
   if (vvDebugMsg::isActive(3))
      cerr << "addPin: type: " << type << " - Position: " << xPos << "...";
   int old_length=pin_data_length;
   unsigned char *temp_array;
   temp_array = new unsigned char[pin_data_length];
   for( i=0; i<pin_data_length; i++)
      temp_array[i]=pin_data[i];
   delete[] pin_data;

   pin_data_length += ( 22 + int(strlen(description)) );
  pin_data = new unsigned char[pin_data_length];
  for( i=0; i<old_length; i++ )
  pin_data[i]=temp_array[i];
  delete[] temp_array;

  number_of_pins++;
  int2octals( &a1, &a2, &a3, &a4, number_of_pins );
  if( a1!=0 || a2!=0 )
  cerr << "## Error: addPin: trying to save more than 255 pins			[ FAILED ]" << endl;
  pin_data[0]=a3; pin_data[1]=a4;
  int descr_length, marker;
  descr_length = int(strlen(description)%256);
  for( i=0; i<descr_length; i++)
  pin_data[old_length+1+i]=description[i];
  marker=old_length+descr_length;
  pin_data[marker+1]=type;
  float2octals( &pin_data[marker+ 2], &pin_data[marker+ 3], &pin_data[marker+ 4], &pin_data[marker+ 5], value1 );
  float2octals( &pin_data[marker+ 6], &pin_data[marker+ 7], &pin_data[marker+ 8], &pin_data[marker+ 9], value2 );
  float2octals( &pin_data[marker+10], &pin_data[marker+11], &pin_data[marker+12], &pin_data[marker+13], value3 );
  float2octals( &pin_data[marker+14], &pin_data[marker+15], &pin_data[marker+16], &pin_data[marker+17], value4 );
  float2octals( &pin_data[marker+18], &pin_data[marker+19], &pin_data[marker+20], &pin_data[marker+21], xPos );
  pin_data[old_length+18]=int(strlen(description)%256);

  if (vvDebugMsg::isActive(3))
  cerr << "					[ OK ]" << endl;
  */
  return errnmb;
}

/// @return 0 if on success, 1 if desired pin was not found
int vvvffile::getPin(int list_number, int number, int* type, float* value1, float* value2, float* value3, float* xPos, int* descr_length)
{
  int jump_len;
  int saved_pins;
  if (!FILEREAD)
    readFile();

  fseek(dz, 0, SEEK_SET);

  fseek(dz, int(headersize + 2), SEEK_CUR);       // header+checksum
  fseek(dz, int(iconsize), SEEK_CUR);             // icondata
  fseek(dz, int(2 + textlength), SEEK_CUR);       // text-length+text+checkbit
  if (fgetc(dz) == 66)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "getPin: Checkbit 1 found						[ OK ]" << endl;
  }
  else
  {
    cerr << "## Error: getPin: Checkbit 1 not correct				[ FAILED ]" << endl;
  }

  // jump over data
  for (i = 0; i < number_of_data_arrays; i++)
  {
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    a3 = (unsigned char)fgetc(dz);
    a4 = (unsigned char)fgetc(dz);
    data_array_length = octals2int(a1, a2, a3, a4);
    fseek(dz, int(data_array_length), SEEK_CUR);
  }

  if (vvDebugMsg::isActive(3))
  {
    cerr << "getPin: Jumped over data: " << number_of_data_arrays << " array(s), " << int(data_array_length) << " bytes			[ OK ]" << endl;
  }

  /* 
   for( i=0; i<number_of_data_arrays; i++ )
   {
      a1=fgetc(dz); a2=fgetc(dz); a3=fgetc(dz); a4=fgetc(dz);
      fseek( dz, int( octals2int( a1, a2, a3, a4 ) ), SEEK_CUR );
   }
   if (vvDebugMsg::isActive(3))
      cerr << "						[ OK ]" << endl;
  */
  if (fgetc(dz) == 99)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "getPin: Checkbit 2 found						[ OK ]" << endl;
  }
  else
  {
    cerr << "## Error: getPin: Checkbit 2 not correct				[ FAILED ]" << endl;
  }

  fseek(dz, 29L, SEEK_CUR);                       // tail-entries to first channel

  if (vvDebugMsg::isActive(3))
    cerr << "getPin: jumping over " << int(number_of_channels) << " channels ...";
  for (i = 0; i < number_of_channels; i++)
  {
    a1 = (unsigned char)fgetc(dz);
    fseek(dz, int(a1 - 1), SEEK_CUR);             // description text
    fseek(dz, int(263), SEEK_CUR);                // rgba-data
  }

  if (vvDebugMsg::isActive(3))
    cerr << "					[ OK ]" << endl;

  // for( i=0; i<100; i++ )
  //    cerr << int(i) << ": " << int(fgetc(dz)) << endl;
  fseek(dz, 1, SEEK_CUR);                         // number of pin lists
  if (vvDebugMsg::isActive(3))
    cerr << "getPin: seeking pin list of interest (number: " << int(list_number) << ") ...";
  for (i = 1; i < list_number; i++)
  {
    fseek(dz, 1, SEEK_CUR);                       // list number
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);

    // cerr << "ZZZ: " << int(a1) << " - " << int(a2) << endl;
    jump_len = (octals2int(0, 0, a1, a2) * 17);
    a1 = (unsigned char)fgetc(dz);
    fseek(dz, int(a1), SEEK_CUR);                 // description text
    fseek(dz, int(jump_len), SEEK_CUR);           // pins
  }

  if (vvDebugMsg::isActive(3))
    cerr << "			[ OK ]" << endl;

  if (list_number != fgetc(dz))
  {
    cerr << "## Error: getPin: could not find correct pin list number " << int(list_number) << "		[ FAILED ]" << endl;
    return 1;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "getPin: found correct pin list						[ OK ]" << endl;

  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  saved_pins = octals2int(0, 0, a1, a2);
  if (number > saved_pins)
  {
    cerr << "## Error: getPin: trying to get pin number " << int(number) << " out of just " << int(saved_pins) << "		[ FAILED]" << endl;
    return 1;
  }

  *descr_length = int(fgetc(dz));
  if (vvDebugMsg::isActive(3))
    cerr << "getPin: length of description text: " << int(*descr_length) << "					[ OK ]" << endl;
  fseek(dz, *descr_length, SEEK_CUR);             // description text

  fseek(dz, ((number - 1) * 17), SEEK_CUR);

  a1 = (unsigned char)fgetc(dz);
  *type = int(a1);
  if (vvDebugMsg::isActive(3))
    cerr << "getPin: getting type of pin: " << int(*type) << "						[ OK ]" << endl;
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  *value1 = octals2float(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "getPin: getting first float: " << float(*value1) << "					[ OK ]" << endl;
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  *value2 = octals2float(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "getPin: getting second float: " << float(*value2) << "					[ OK ]" << endl;
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  *value3 = octals2float(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "getPin: getting third float: " << float(*value3) << "					[ OK ]" << endl;
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  a3 = (unsigned char)fgetc(dz);
  a4 = (unsigned char)fgetc(dz);
  *xPos = octals2float(a1, a2, a3, a4);
  if (vvDebugMsg::isActive(3))
    cerr << "getPin: getting x-position: " << float(*xPos) << "					[ OK ]" << endl;

  return 0;
}

int vvvffile::getPinDescription(char*)
{
  cerr << "# Warning: getPinDescripition: function not yet implemented			[ WRN ]" << endl;
  return 1;
}

int vvvffile::initTFChannel(void)
{
  errnmb = 0;
  if (tf_channel)
    delete[] tf_channel;
  tf_channel = new unsigned char[256];
  for (i = 0; i < 256; i++)
    tf_channel[i] = 0;
  number_of_tf_points = 0;
  return errnmb;
}

int vvvffile::addTFPoint(int position, int value)
{
  errnmb = 0;
  if (number_of_tf_points > 127)
  {
    cerr << "## Error: addTFPoint: Existing TF-Point-Channel is full, save it and create next" << endl;
    errnmb = 1;
    return errnmb;
  }
  else
  {
    if (position < 256 && value < 256)
    {
      number_of_tf_points++;
      tf_channel[position * 2] = (unsigned char)position;
      tf_channel[position * 2 + 1] = (unsigned char)value;
      return 0;
    }
    else
    {
      cerr << "## Error: addTFPoint was given wrong parameters	[ FAILED ]" << endl;
      errnmb = 1;
      return errnmb;
    }
  }
}

int vvvffile::delTFPoint(int position)
{
  errnmb = 0;
  if (position > 128)
  {
    cerr << "## Error: delTFPoint was given wrong parameters		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else
  {
    for (i = position; i < number_of_tf_points; i++)
    {
      tf_channel[i * 2] = tf_channel[i * 2 + 1];
    }

    number_of_tf_points--;
    for (i = number_of_tf_points; i < 129; i++)
    {
      tf_channel[i * 2] = 0;
      tf_channel[i * 2 + 1] = 0;
    }

    tf_channel[255] = 0;
    return errnmb;
  }
}

int vvvffile::getTFPoints(int number, unsigned char* tf_channel, char* channel_description)
{
  int jump_width;
  errnmb = 0;

  // see getRGBAarray, similar, same file
  if (!FILEREAD)
    readFile();
  if (number > number_of_tf_channels)
  {
    cerr << "## Error: trying to get tf-channel " << number << " out of just " << number_of_tf_channels << endl;
    errnmb = 1;
    return errnmb;
  }

  if (number < 1)
    cerr << "# Warning: File has no tf-channels, do not try to get one			[ WRN ]" << endl;
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "getTFPoints: open file for LOADing: " << fileName << "[ OK ]" << endl;

  fseek(dz, int(headersize + 2), SEEK_CUR);       // header+checksum
  fseek(dz, int(iconsize), SEEK_CUR);             // icondata
  fseek(dz, int(2 + textlength), SEEK_CUR);       // text-length+text+checkbit
  if (fgetc(dz) == 66)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "getTFPoints: Checkbit 1 found					[ OK ]" << endl;
    else
    {
      cerr << "## Error: getTFPoints: Checkbit 1 not correct		[ FAILED ]" << endl;
      errnmb = 1;
      return errnmb;
    }
  }

  if (vvDebugMsg::isActive(3))
    cerr << "getTFPoints: jumping over data ..";
  for (i = 0; i < number_of_data_arrays; i++)
  {
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    a3 = (unsigned char)fgetc(dz);
    a4 = (unsigned char)fgetc(dz);
    fseek(dz, int(octals2int(a1, a2, a3, a4)), SEEK_CUR);
  }

  if (vvDebugMsg::isActive(3))
    cerr << "					[ OK ]" << endl;
  if (fgetc(dz) == 99)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "getTFPoints: Checkbit 2 found					[ OK ]" << endl;
    else
    {
      cerr << "## Error: getTFPoints: Checkbit 2 not correct		[ FAILED ]" << endl;
      errnmb = 1;
      return errnmb;
    }
  }

  fseek(dz, 29L, SEEK_CUR);                       // tail-entries to first channel

  i = 0;
  {
    jump_width = 0;

    a1 = (unsigned char)fgetc(dz);
    fseek(dz, int(a1), SEEK_CUR);                 // description text
    a1 = (unsigned char)fgetc(dz);
    jump_width += int(a1) + 2;
    if (a1 > 4 && a1 < 9)
      i++;
  }

  while (i < number)
    fseek(dz, (int(jump_width) * (-1)), SEEK_CUR);

  a1 = (unsigned char)fgetc(dz);
  if (channel_description)
    delete[] channel_description;
  channel_description = new char[int(a1) + 1];

  // SUCHMARKE
  if (channel_description)
    delete[] channel_description;

  fseek(dz, int(a1 - 1), SEEK_CUR);
  fseek(dz, 7L, SEEK_CUR);
  size_t retval;
  retval=fread(tf_channel, 1, 256, dz);
  if (retval!=256)
  {
    std::cerr<<"vvvffile::getTFPoints: fread failed"<<std::endl;
    return 1;
  }

  return errnmb;
}

int vvvffile::saveTFPoints(unsigned char* tf_channel, char* description, int type)
{
  int             jumper, old_length, new_length, description_length;
  unsigned char*  tempchannel;
  errnmb = 0;

  // see addRGBAarray, similar, same output-file
  // prepare everything needed
  description_length = strlen(description);
  old_length = channel_array_length;
  new_length = old_length + 263 + description_length;
  jumper = old_length;

  // enlarge channel to place further rgba's in it
  tempchannel = new unsigned char[old_length];
  for (i = 0; i < old_length; i++)
    tempchannel[i] = channel[i];
  delete[] channel;
  channel = new unsigned char[new_length];
  for (i = 0; i < old_length; i++)
    channel[i] = tempchannel[i];
  delete[] tempchannel;

  // length of channel description
  channel[jumper++] = 0;

  // type of channel
  channel[jumper++] = (unsigned char)(type);

  // describing which voxel-bytes
  channel[jumper++] = (unsigned char)(type - 4);

  // x-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // y-axis values, 0,255=255
  channel[jumper++] = 0;
  channel[jumper++] = 255;

  // data of tf channel
  for (i = 0; i < 256; i++)
    channel[jumper + i] = tf_channel[i];
  jumper += 256;

  return errnmb;
}

int vvvffile::delTFChannel(void)
{
  errnmb = 0;
  if (tf_channel)
    delete[] tf_channel;
  return errnmb;
}

int vvvffile::initChannels(void)
{
  errnmb = 0;
  if (channel)
    delete[] channel;
  number_of_channels = 0;
  number_of_rgba_channels = 0;
  number_of_tf_channels = 0;
  channel_array_length = 1;
  channel = new unsigned char[channel_array_length];
  channel[0] = (unsigned char)number_of_channels;
  return errnmb;
}

int vvvffile::delChannels(void)
{
  errnmb = 0;
  if (channel)
    delete[] channel;
  number_of_channels = 0;
  number_of_rgba_channels = 0;
  number_of_tf_channels = 0;
  channel_array_length = 1;
  return errnmb;
}

int vvvffile::getDataArraySize(int number)
{
  int size;
  if (!FILEREAD)
    readFile();
  if (number > number_of_data_arrays || number < 0)
  {
    cerr << "## Error: trying to get data array " << number << " but only " << number_of_data_arrays << " present		[ FAILED ]" << endl;
  }

  size = voxel_number_x * voxel_number_y * voxel_number_z * bytes_per_voxel;
  if (number == 0)
    size *= number_of_data_arrays;
  return size;
}

int vvvffile::findBestTypeOfCoding(unsigned char* data, int* coding_parameter, int* best_length)
{
  int   best_type_of_coding = 0;
  int   current_length = 0;

  int*  counter_array, rle_parameter;
  int   counter, uncoded_length, current_value, marker;

  // No Encoding 0
  *best_length = bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z;
  best_type_of_coding = 0;
  *coding_parameter = 0;

  // Run-Length-Encoding (RLE) type of coding 1
  counter_array = new int[256];
  for (i = 0; i < 256; i++)
    counter_array[i] = 0;
  for (i = 0; i < (bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z); i++)
    counter_array[data[i]] += 1;

  j = counter_array[0];
  rle_parameter = counter_array[0];
  k = 0;
  for (i = 0; i < 256; i++)
    if (counter_array[i] < j)
  {
    j = counter_array[i];
    rle_parameter = i;
  }

  current_length = 1;
  j = 0;
  i = 0;

  uncoded_length = bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z;
  current_value = -1;
  current_length = 1;
  marker = 0;
  do
  {
    current_value = data[marker];
    counter = 0;
    do
    {
      counter += 1;
      marker++;
    } while (marker < uncoded_length && data[marker] == data[marker - 1] && counter < 255);
    if (counter > 3)
    {
      current_length += 3;
    }
    else
    {
      if (current_value == rle_parameter)
      {
        current_length += counter * 3;
      }
      else
      {
        current_length += counter;
      }
    }
  } while (marker < uncoded_length);

  //   cerr << "current_length: " << int( current_length ) << endl;
  /* 
   do
   {
      j=data[i];
     counter=0;
      do
      {
         counter+=1;
         i++;
      }while( i<( bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z ) && data[i]==data[i-1] && (counter<65536) );
     if( counter>3 )
  {
  current_length += 4;
  }
  else
  {
  if( j==rle_parameter )
  {
  current_length += counter*4;
  }
  else
  {
  current_length += counter;
  }
  }

  }while ( i<( bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z ) ); */
  if (current_length < (*best_length))
  {
    *best_length = current_length;
    best_type_of_coding = 1;                      // RLE
    *coding_parameter = rle_parameter;            // what value is used to mark RLE
  }

  /*
  // ZIP 2
  //datei oeffnen
  fseek(datei, 0L, SEEK_END);
  laenge = ftell(datei);
  fseek(datei, 0L, SEEK_SET);
  // jetzt biste wieder am Dateianfang
  */
  // ZIP encoding, type 2
  // Byte *compr;
  // uLong comprLen;
  /*    Byte *compr;
    uLong comprLen;
  cerr << int(uncoded_length) << endl;
  int err;
  cerr << "ZIP: " <<  "blubb" << endl;
  err = compress(compr, &comprLen, (const Bytef*)data, uncoded_length);
  cerr << "ZIP: " <<  "blubb" << endl;
  cerr << "ZIP: " << int(comprLen) << " - " << int(compr) << endl;
  */
  if (vvDebugMsg::isActive(3))
    cerr << "findBestTypeOfCoding: ";
  switch (best_type_of_coding)
  {
    case 0:
      if (vvDebugMsg::isActive(3))
        cerr << "unencoded						[ OK ]" << endl;
      break;
    case 1:
      if (vvDebugMsg::isActive(3))
      {
        cerr << "runlength encoded: " << (*best_length * 100) / (bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z) << "%				[ OK ]" << endl;
      }
      break;
    default:
      if (vvDebugMsg::isActive(3))
        cerr << "## Error: unrecognized coding					[ OK ]" << endl;
      break;
  }

  delete[] counter_array;
  return best_type_of_coding;
}

int vvvffile::readDataArray(unsigned char ** daten, int array_number)
{
  errnmb = 0;
  if (!FILEREAD)
    readFile();
  if (array_number > number_of_data_arrays)
  {
    cerr << "## Error: trying to get data_array " << array_number << " out of just " << number_of_data_arrays << endl;
    errnmb = 1;
    return errnmb;
  }

  if (array_number < 0)
    cerr << "# Warning: trying to get data_array entry number " << array_number << "		[ WRN ]" << endl;
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }
  else if (vvDebugMsg::isActive(3))
    cerr << "readDataArray: open file for LOADing: " << fileName << "		[ OK ]" << endl;

  fseek(dz, int(headersize + 2), SEEK_CUR);       // header+checksum
  fseek(dz, int(iconsize), SEEK_CUR);             // icondata
  fseek(dz, int(2 + textlength), SEEK_CUR);       // text-length+text+checkbit
  if (fgetc(dz) == 66)
    if (vvDebugMsg::isActive(3))
      cerr << "readDataArray: Checkbit 1 found						[ OK ]" << endl;
  else
  {
  }
  else
  {
    cerr << "## Error: readDataArray: Checkbit 1 incorrect		[ FAILED ]" << endl;
    errnmb = 1;
    return errnmb;
  }

  if (array_number != 0)
  {
    unsigned char*  blubbs;
    int             laengens, kodierung;
    if (array_number != 1)
    {
      if (vvDebugMsg::isActive(3))
        cerr << "readDataArray: jumping to data_array of interest (" << int(array_number) << ")";
      for (i = 1; i < array_number; i++)
      {
        a1 = (unsigned char)fgetc(dz);
        a2 = (unsigned char)fgetc(dz);
        a3 = (unsigned char)fgetc(dz);
        a4 = (unsigned char)fgetc(dz);
        fseek(dz, int(octals2int(a1, a2, a3, a4)), SEEK_CUR);
      }

      if (vvDebugMsg::isActive(3))
        cerr << "			[ OK ]" << endl;
    }

    data_array = new unsigned char[bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z];
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    a3 = (unsigned char)fgetc(dz);
    a4 = (unsigned char)fgetc(dz);
    laengens = int(octals2int(a1, a2, a3, a4) - 5);
    a1 = (unsigned char)fgetc(dz);
    a2 = (unsigned char)fgetc(dz);
    a3 = (unsigned char)fgetc(dz);
    a4 = (unsigned char)fgetc(dz);
    kodierung = int(fgetc(dz));
    blubbs = new unsigned char[laengens];

    int retval;
    retval=fread(blubbs, 1, laengens, dz);
    if (retval!=laengens)
    {
      std::cerr<<"vvvffile::readDataArray: fread failed"<<std::endl;
      delete[] blubbs;
      return 1;
    }

    //   for( i=0; i<laengens; i++ )
    //   {
    //      blubbs[i]=fgetc(dz);
    //   }
    decodeData(blubbs, laengens, kodierung, data_array);
    *daten = data_array;
    delete[] blubbs;
  }
  else
  {
    // read all data_arrays
    cerr << "## Error: FUNCTION TO READ ALL DATA ARRAYS IN SINGLE OPERATION IS NOT YET IMPLEMENTED" << endl;
    errnmb = 1;
    return errnmb;
  }

  if (fgetc(dz) == 99)
  {
    if (vvDebugMsg::isActive(3))
      cerr << "readDataArray: Checkbit 2 found						[ OK ]" << endl;
    else
    {
      cerr << "## Error: getRGBAChannel: Checkbit 2 not correct		[ FAILED ]" << endl;
      errnmb = 1;
      return errnmb;
    }
  }

  return errnmb;
}

int vvvffile::giveUpDataArray(void)
{
  errnmb = 0;
  if (data_array)
    delete[] data_array;
  return errnmb;
}

int vvvffile::mbit(float* value)
{
  if (*value >= 1)
  {
    *value -= 1;
    *value *= 2;
    return 1;
  }
  else
  {
    *value *= 2;
    return 0;
  }
}

void vvvffile::int2octals(unsigned char* a1, unsigned char* a2, unsigned char* a3, unsigned char* a4, int value)
{
  *a1 = (unsigned char)((value & (255 << 24)) >> 24);
  *a2 = (unsigned char)((value & (255 << 16)) >> 16);
  *a3 = (unsigned char)((value & (255 << 8)) >> 8);
  *a4 = (unsigned char)(value & (255));
}

int vvvffile::octals2int(unsigned char a1, unsigned char a2, unsigned char a3, unsigned char a4)
{
  int value;
  value = a1 * 16777216 + a2 * 65536 + a3 * 256 + a4;
  return value;
}

float vvvffile::octals2float(unsigned char a1, unsigned char a2, unsigned char a3, unsigned char a4)
{
  float Wert, Mantissenwert;
  int   Vorzeichen, Exponent, Mantisse;
  int   i;

  if (a1 >= 128)
  {
    a1 -= 128;
    Vorzeichen = -1;
  }
  else
  {
    Vorzeichen = +1;
  }

  Exponent = a1 << 1;
  if (a2 >= 128)
  {
    Exponent += 1;
    a2 -= 128;
  }

  Exponent -= 128;                                // Offset, Exponent given in SM-Format

  Mantisse = int(a2 << 16) + int(a3 << 8) + int(a4);

  Mantissenwert = 0;
  for (i = 1; i < 24; i++)
    Mantissenwert += float(pow((float)2, -(23 - i))) * ((Mantisse & (1 << (i - 1))) >> (i - 1));

  // Zusammensetzen
  Wert = float(Vorzeichen) * float(1 + Mantissenwert) * float(pow((float)2, Exponent));

  return Wert;
}                                                 // end of octals2float

void vvvffile::float2octals(unsigned char* a1, unsigned char* a2, unsigned char* a3, unsigned char* a4, float value)
{
  int Exponent;
  *a1 = *a2 = *a3 = *a4 = 0;

  if (value == 0)                                 // Null-Check
  {
    *a1 = *a2 = *a3 = *a4 = 0;
  }
  else
  {
    // Vorzeichen
    if (value > 0)
    {
      *a1 = 0;
    }
    else
    {
      value = value * (-1);
      *a1 = 128;
    }

    // Exponent
    Exponent = 0;
    if (value >= 2)
    {
      do
      {
        value /= 2;
        Exponent += 1;
      } while (value >= 2);
    }
    else
    {
      do
      {
        value *= 2;
        Exponent -= 1;
      } while (value < 1);
    }

    Exponent += 128;                              // Offset
    *a2 = (unsigned char)(128 * (Exponent % 2));
    Exponent = int(Exponent / 2);
    *a1 = (unsigned char)(*a1 + Exponent);

    // Mantisse
    value -= 1;
    *a2 = (unsigned char)(*a2 + (64 * mbit(&value)));
    *a2 = (unsigned char)(*a2 + (32 * mbit(&value)));
    *a2 = (unsigned char)(*a2 + (16 * mbit(&value)));
    *a2 = (unsigned char)(*a2 + (8 * mbit(&value)));
    *a2 = (unsigned char)(*a2 + (4 * mbit(&value)));
    *a2 = (unsigned char)(*a2 + (2 * mbit(&value)));
    *a2 = (unsigned char)(*a2 + (1 * mbit(&value)));

    *a3 = (unsigned char)(*a3 + (128 * mbit(&value)));
    *a3 = (unsigned char)(*a3 + (64 * mbit(&value)));
    *a3 = (unsigned char)(*a3 + (32 * mbit(&value)));
    *a3 = (unsigned char)(*a3 + (16 * mbit(&value)));
    *a3 = (unsigned char)(*a3 + (8 * mbit(&value)));
    *a3 = (unsigned char)(*a3 + (4 * mbit(&value)));
    *a3 = (unsigned char)(*a3 + (2 * mbit(&value)));
    *a3 = (unsigned char)(*a3 + (1 * mbit(&value)));

    *a4 = (unsigned char)(*a4 + (128 * mbit(&value)));
    *a4 = (unsigned char)(*a4 + (64 * mbit(&value)));
    *a4 = (unsigned char)(*a4 + (32 * mbit(&value)));
    *a4 = (unsigned char)(*a4 + (16 * mbit(&value)));
    *a4 = (unsigned char)(*a4 + (8 * mbit(&value)));
    *a4 = (unsigned char)(*a4 + (4 * mbit(&value)));
    *a4 = (unsigned char)(*a4 + (2 * mbit(&value)));
    *a4 = (unsigned char)(*a4 + (1 * mbit(&value)));
  }                                               // enf of Null-Check
}                                                 // end of float2octals

void vvvffile::encodeData(unsigned char* daten, int code, int para, unsigned char* coded_data)
{
  switch (code)
  {
    case 0:
      // do not encode data
      if (vvDebugMsg::isActive(3))
        cerr << "encodeData: do not encode data 						[ OK ]" << endl;
      for (i = 0; i < (voxel_number_x * voxel_number_y * voxel_number_z * bytes_per_voxel); i++)
      {
        coded_data[i] = daten[i];
      }
      break;
    case 1:
      // RLE - Run Length Encoding
      int counter, coded_marker, uncoded_marker, current_value, uncoded_length, k;
      uncoded_length = bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z;
      uncoded_marker = 0;
      counter = 0;
      coded_data[0] = (unsigned char)para;
      coded_marker = 1;
      if (vvDebugMsg::isActive(3))
        cerr << "encodeData: runlength encoding";

      do
      {
        current_value = daten[uncoded_marker];
        counter = 0;
        do
        {
          counter += 1;
          uncoded_marker += 1;
        } while (uncoded_marker < uncoded_length && daten[uncoded_marker] == daten[uncoded_marker - 1] && counter < 255);
        if (counter > 2)
        {
          coded_data[coded_marker + 0] = (unsigned char)para;
          coded_data[coded_marker + 1] = (unsigned char)current_value;
          coded_data[coded_marker + 2] = (unsigned char)counter;
          coded_marker += 3;
        }
        else
        {
          if (current_value == para)
          {
            for (k = 0; k < counter; k++)
            {
              coded_data[coded_marker + 0] = (unsigned char)para;
              coded_data[coded_marker + 1] = (unsigned char)current_value;
              coded_data[coded_marker + 2] = (unsigned char)counter;
              coded_marker += 3;
            }
          }
          else
          {
            for (k = 0; k < counter; k++)
            {
              coded_data[coded_marker] = (unsigned char)current_value;
              coded_marker += 1;
            }
          }
        }
      } while (uncoded_marker < uncoded_length);
      if (vvDebugMsg::isActive(3))
        cerr << " 				 		[ OK ]" << endl;

      //	       for(k=0; k<uncoded_length; k++){cerr << int(k) << ": " << int(coded_data[k]) << " - " << int(daten[k]) << endl;}
      break;                                      // end RLE   */

      /*	     // RLE
              if (vvDebugMsg::isActive(3))
              cerr << "encodeData: runlength encoding";
         int counter, marker, k;
         coded_data[0]=para;
         marker=1;
           i=0;
         do
           {
              j=daten[i];
              counter=0;
      do
      {
      counter+=1;
      i+=1;
      }while( daten[i]==daten[i-1] && (counter<256) && i<( bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z ) );
      if( counter>2 )
      {
      coded_data[marker+0]=para;
      coded_data[marker+1]=j;
      coded_data[marker+2]=counter;
      marker += 3;
      }
      else
      {
      if( j==para )
      {
      for( k=0; k<counter; k++ )
      {
      coded_data[marker+0]=para;
      coded_data[marker+1]=j;
      coded_data[marker+2]=1;
      marker+=3;
      }
      }
      else
      {
      for( k=0; k<counter; k++ )
      {
      coded_data[marker+0]=j;
      marker+=1;
      }
      }
      }
      }while ( i<( bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z ) );
      cerr << "ENCODE: " << int( marker ) << endl;
      if (vvDebugMsg::isActive(3))
      cerr << "						[ OK ]" << endl;
      break; // end RLE   */
    default:
      // NOT SUPPORTED TYPE - ERROR
      if (vvDebugMsg::isActive(3))
        cerr << "## Error: requested type of coding " << int(code) << " unknown to encodeData			[FAILED]" << endl;
      break;
  }

  //for( i=0; i<100; i++ ) cerr << int(i) << ": " << int(daten[i]) << " - " << int(coded_data[i]) << endl;
}

void vvvffile::decodeData(unsigned char* daten, int rle_length, int code, unsigned char* decoded_data)
{
  switch (code)
  {
    case 0:
      // do not decode data
      if (vvDebugMsg::isActive(3))
        cerr << "decodeData: not encoded data						[ OK ]" << endl;
      for (i = 0; i < (voxel_number_x * voxel_number_y * voxel_number_z * bytes_per_voxel); i++)
      {
        decoded_data[i] = daten[i];
      }
      break;
    case 1:
      if (vvDebugMsg::isActive(3))
        cerr << "decodeData: decoding rle-encoded data";

      int rle_parameter, coded_marker, uncoded_marker;
      uncoded_marker = 0;
      coded_marker = 0;

      rle_parameter = daten[coded_marker];
      coded_marker += 1;
      do
      {
        if (daten[coded_marker] == rle_parameter)
        {
          for (j = 0; j < daten[coded_marker + 2]; j++)
          {
            decoded_data[uncoded_marker] = daten[coded_marker + 1];
            uncoded_marker += 1;
          }

          coded_marker += 3;
        }
        else
        {
          decoded_data[uncoded_marker] = daten[coded_marker];
          uncoded_marker += 1;
          coded_marker += 1;
        }
      } while (coded_marker < rle_length);
      if (vvDebugMsg::isActive(3))
        cerr << "					[ OK ]" << endl;
      break;

      /*            if (vvDebugMsg::isActive(3))
              cerr << "decodeData: decoding rle-encoded data";
           int rle_parameter, marker;
         marker=0;
         i=1;
         rle_parameter=daten[0];
           do
         {
            if( daten[i]==rle_parameter )
           {
              for( j=0; j<daten[i+2]; j++ )
      {
      decoded_data[marker]=daten[i+1];
      marker+=1;
      }
      i+=3;
      }
      else
      {
      decoded_data[marker]=daten[i];
      marker+=1;
      i+=1;
      }
      }while( i<rle_length );
      if (vvDebugMsg::isActive(3))
      cerr << "					[ OK ]" << endl;
      break;*/
    default:
      if (vvDebugMsg::isActive(3))
        cerr << "## Error: decodeData: unrecognized type of coding (" << int(code) << ")			[FAILED]" << endl;
      break;
  }
}

int vvvffile::readHeaderNtBoxBox(void)
{
  int   N, i, errnmb;
  float t, x1, y1, z1, x2, y2, z2;
  char  str[1024];
  errnmb = 0;

  // Just give the wanted dimension of the cube
  // resolve volume dimensions X
  voxel_number_x = 128;

  // resolve volume dimensions Y
  voxel_number_y = 128;

  // resolve volume dimensions Z
  voxel_number_z = 128;

  // resolve volume dimensions Z
  bytes_per_voxel = 1;

  // channels not saved
  channel_array_length = 1;

  // Open file
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error 101: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 101;                                 // Datei nicht vorhanden oder fehlerhaft
    return errnmb;
  }

  // Count frames
  cerr << "readHeaderNtBoxBox: Counting frames: ";
  frames_number = 0;
  while (!feof(dz))
  {
    ++frames_number;
    int retval;
    retval=fscanf(dz, "%d%f%f%f%f%f%f%f", &N, &t, &x1, &y1, &z1, &x2, &y2, &z2);
    if (retval!=8)
    {
      std::cerr<<"vvvffile::readHeaderNtBoxBox: fscanf failed"<<std::endl;
      return 1;
    }
    for (i = 0; i < N; i++)
    {
      char* retval;
      retval=fgets(str, 1023, dz);
      if (retval==NULL)
      {
        std::cerr<<"vvvffile::readHeaderNtBoxBox: fgets failed"<<std::endl;
        return 1;
      }
    }
  }

  cerr << frames_number << "					[ OK ]" << endl;
  fclose(dz);
  return errnmb;
}

int vvvffile::readDataxyzvxvyvzrnc(unsigned char* data, int c)
{
  int     N;
  double  xyz;
  float   t, x1, y1, z1, x2, y2, z2;
  float   xx, yy, zz, vx, vy, vz, ra, nc;

  //   char str[1024];
  // datei oeffnen und an anfang der daten springen
  // Open file
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error 101: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 101;                                 // Datei nicht vorhanden oder fehlerhaft
    return errnmb;
  }

  // initialise data array
  if (vvDebugMsg::isActive(3))
    cerr << "readASCII: Setting all values in data-array to 0:";
  data_array_length = voxel_number_x * voxel_number_y * voxel_number_z * bytes_per_voxel;
  for (i = 0; i < data_array_length; i++)
    data[i] = 0;
  if (vvDebugMsg::isActive(3))
    cerr << "			[ OK ]" << endl;

  // jump to intersting values
  i = 1;
  if (c > frames_number)
  {
    cerr << "## Error: requesting data array " << int(c) << ", but " << fileName << " has only " << frames_number << "		[ FAILED ]" << endl;
  }

  if (vvDebugMsg::isActive(3))
    cerr << "readASCII: jumping to interesting part ";
  while (i != c)
  {
    char  str[1024];
    if (vvDebugMsg::isActive(3))
      cerr << ".";
    i++;

    // resolve number of particles
    int retval;
    retval=fscanf(dz, "%d%f%f%f%f%f%f%f", &N, &t, &x1, &y1, &z1, &x2, &y2, &z2);
    if (retval!=8)
    {
      std::cerr<<"vvvffile::readDataxyzvxvyvzrnc: fscanf failed"<<std::endl;
      return 1;
    }
    for (j = 0; j < (N + 1); j++)
    {
      char* retval;
      retval=fgets(str, 1023, dz);
      if (retval==NULL)
      {
        std::cerr<<"vvvffile::readDataxyzvxvyvzrnc: fgets failed"<<std::endl;
        return 1;
      }
    }

    //	  for( j=0; j<N; j++ )
    //         (void) fscanf( dz, "%f%f%f%f%f%f%f%f", &xx, &yy, &zz, &vx, &vy, &vz, &ra, &nc );
  }

  if (vvDebugMsg::isActive(3))
    cerr << "   				[ OK ]" << endl;

  // reading data
  int retval;
  retval=fscanf(dz, "%d%f%f%f%f%f%f%f", &N, &t, &x1, &y1, &z1, &x2, &y2, &z2);
  if (retval!=1)
  {
    std::cerr<<"vvvffile::readDataxyzvxvyvzrnc: fscanf failed"<<std::endl;
    return 1;
  }
  time_stamp = int(t);
  if (vvDebugMsg::isActive(3))
  {
    cerr << "readASCII: Number of particles " << N << " at timestamp " << t << "			[ OK ]" << endl;
    cerr << "readASCII: X-Range from " << x1 << " to " << x2 << "					[ OK ]" << endl;
    cerr << "readASCII: Y-Range from " << y1 << " to " << y2 << "					[ OK ]" << endl;
    cerr << "readASCII: Z-Range from " << z1 << " to " << z2 << "					[ OK ]" << endl;
  }

  // set real dimensions of voxels
  realsize_x = float((x2 - x1) / voxel_number_x);
  realsize_y = float((y2 - y1) / voxel_number_y);
  realsize_z = float((z2 - z1) / voxel_number_z);

  if (vvDebugMsg::isActive(3))
    cerr << "readASCII: transforming coordinates:";

  for (i = 0; i < N; i++)
  {
    retval=fscanf(dz, "%f%f%f%f%f%f%f%f", &xx, &yy, &zz, &vx, &vy, &vz, &ra, &nc);
    if (retval!=8)
    {
      std::cerr<<"vvvffile::readDataxyzvxvyvzrnc: fscanf failed"<<std::endl;
      return 1;
    }

    // CHECK FOR ERRORS IN DATA
    if (xx < x1 || xx > x2)
      cerr << endl << "X-Error in line " << i << " xx: " << xx << " has to be between " << x1 << " and " << x2 << endl;
    if (yy < y1 || yy > y2)
      cerr << endl << "Y-Error in line " << i << " yy: " << yy << " has to be between " << y1 << " and " << y2 << endl;
    if (zz < z1 || zz > z2)
      cerr << endl << "Z-Error in line " << i << " zz: " << zz << " has to be between " << z1 << " and " << z2 << endl;

    j = int((xx - x1) / (x2 - x1) * voxel_number_x);
    if (j == voxel_number_x)
      j -= 1;
    xyz = j * voxel_number_y * voxel_number_z;
    j = int((yy - y1) / (y2 - y1) * voxel_number_y);
    if (j == voxel_number_y)
      j -= 1;
    xyz += j * voxel_number_z;
    j = int((zz - z1) / (z2 - z1) * voxel_number_z);
    if (j == voxel_number_z)
      j -= 1;
    xyz += j;
    j = int(xyz);

    //	  xyz = (xx-x1)/(x2-x1) * voxel_number_x * voxel_number_y * voxel_number_z ;
    //	  xyz+= (yy-y1)/(y2-y1) * voxel_number_y * voxel_number_z ;
    //	  xyz+= (zz-z1)/(z2-z1) * voxel_number_z ;
    //	  j=int(xyz);
    // cerr << int(i) << "/" << int(N) << "  	|	" << float((xx-x1)/(x2-x1)) << "  	|	" << float((yy-y1)/(y2-y1)) << "  	|	" << float((zz-z1)/(z2-z1)) << "	- " << int(j) << endl;
    // cerr << int(i) << "/" << int(N) << "	|	" << int(128*128*128-j) << endl;
    data[j] = 255;
  }

  if (vvDebugMsg::isActive(3))
    cerr << "					[ OK ]" << endl;

  fclose(dz);
  return 0;
}

// *************************** for unsupported files -end- *****
void vvvffile::setValues(int vers_numb, int head_size, int temp_numb, char* temp_file)
{
  version_number = vers_numb;
  headersize = head_size;
  temp_array_number = temp_numb;

  // resolve name of tempfile
  //			cerr << "1" << endl;
  if (tempfile != NULL)
  {
    delete[] tempfile;
    tempfile = NULL;
  }

  //			cerr << "2" << endl;
  //			   char *tempfile;
  tempfile = new char[strlen(temp_file) + 2];

  //			cerr << "3" << endl;
  strcpy(tempfile, temp_file);

  //			cerr << "4" << endl;
  for (i = strlen(tempfile); tempfile[i] != '.' && i; i--)
    ;

  //			cerr << "5" << endl;
  tempfile[i + 1] = 't';
  tempfile[i + 2] = 'm';
  tempfile[i + 3] = 'p';
  tempfile[i + 4] = 0;

  //			 cerr << "tempfile: " << temp_file << endl;
  //			 cerr << "TEMPFILE: " << tempfile << endl;
}

// *************************** for unsupported files -start- ***
int vvvffile::readXYZunsupp(void)
{
  int errnmb = 0;

  // Open file
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error 101: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 101;                                 // Datei nicht vorhanden oder fehlerhaft
    return errnmb;
  }

  // resolve volume dimensions X
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  voxel_number_x = octals2int(0, 0, a1, a2);
  if (voxel_number_x == 0)
  {
    cerr << "## Error 104: file corrupt or not a volume data file	[ FAILED ]" << endl;
    errnmb = 104;                                 // angeblich keine x-ausdehnung
  }
  else
  {
    if (vvDebugMsg::isActive(3))
      cerr << "readXYZunsupp: Voxel Number X checked: OK				[ " << voxel_number_x << " ]" << endl;
  }

  // resolve volume dimensions Y
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  voxel_number_y = octals2int(0, 0, a1, a2);
  if (voxel_number_y == 0)
  {
    cerr << "## Error 104: file corrupt or not a volume data file	[ FAILED ]" << endl;
    errnmb = 104;                                 // angeblich keine x-ausdehnung
  }
  else
  {
    if (vvDebugMsg::isActive(3))
      cerr << "readXYZunsupp: Voxel Number Y checked: OK				[ " << voxel_number_y << " ]" << endl;
  }

  // resolve volume dimensions Z
  a1 = (unsigned char)fgetc(dz);
  a2 = (unsigned char)fgetc(dz);
  voxel_number_z = octals2int(0, 0, a1, a2);
  if (voxel_number_z == 0)
  {
    cerr << "## Error 104: file corrupt or not a volume data file	[ FAILED ]" << endl;
    errnmb = 104;                                 // angeblich keine x-ausdehnung
  }
  else
  {
    if (vvDebugMsg::isActive(3))
      cerr << "readXYZunsupp: Voxel Number Z checked: OK				[ " << voxel_number_z << " ]" << endl;
  }

  // Just one byte to describe voxel data
  bytes_per_voxel = 1;

  // real dimensions do not matter but have to be set
  realsize_x = 1.234f;
  realsize_y = 1.234f;
  realsize_z = 1.234f;

  // fclose( dz );
  return errnmb;
}

int vvvffile::readDATAunsupp(unsigned char* data)
{
  int errnmb = 0;

  // datei oeffnen und an anfang der daten springen
  // Open file
  dz = fopen(fileName, "rb");
  if (!dz)
  {
    cerr << "## Error 101: Unable to open file for reading		[ FAILED ]" << endl;
    errnmb = 101;                                 // Datei nicht vorhanden oder fehlerhaft
    return errnmb;
  }

  // jump over x, y, z
  fseek(dz, 6L, SEEK_SET);

  // read the data
  data_array_length = voxel_number_x * voxel_number_y * voxel_number_z * bytes_per_voxel;
  if (errnmb == 0)
  {
    /*
        int counter, l, m, n;
         for( i=0; i<(voxel_number_x); i++ )
         {
            for( j=0; j<(voxel_number_y); j++ )
            {
               for( k=0; k<(voxel_number_z); k++ )
               {
               counter=k*voxel_number_x*voxel_number_y + j*voxel_number_x + i;
              l=counter%int(voxel_number_x*voxel_number_y);
              m=(counter-l*voxel_number_x*voxel_number_y)%int(voxel_number_x);
    n=counter - l*voxel_number_x*voxel_number_y - m*voxel_number_x;
    data[counter]=(unsigned char)(fgetc(dz)&0x0FF);
    if( (i==0)&&(j==0) ) cerr << i << " " << j << " " << k << " -> " << counter << " -> " << l << " " << m << " " << n << endl;
    }
    }
    }
    */
    int retval;
    retval=fread(data, 1, data_array_length, dz);
    if (retval!=data_array_length)
    {
      std::cerr<<"vvvffile::readDATAunsupp: fread failed"<<std::endl;
      return 1;
    }

    //         for( i=0; i<data_array_length; i++ )
    //         {
    //            data[i]=(unsigned char)(fgetc(dz)&0x0FF);
    //         }
  }
  else
  {
    cerr << "Correct start of data could not be found....aborting" << endl;
  }

  if (vvDebugMsg::isActive(3))
    cerr << "readDATAunsupp: read file-data: " << data_array_length << " bytes				[ OK ]" << endl << endl;

  fclose(dz);
  return 0;
}

int vvvffile::getDataSize(void)
{
  // resolve size of required data array
  return bytes_per_voxel * voxel_number_x * voxel_number_y * voxel_number_z;
}

void vvvffile::getResolution(int* x, int* y, int* z, int* b)
{
  *x = voxel_number_x;
  *y = voxel_number_y;
  *z = voxel_number_z;
  *b = bytes_per_voxel;
  return;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
