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

#ifndef VV_VFFILE_H
#define VV_VFFILE_H

#include "vvplatform.h"
#include <stdio.h>
#include <string.h>

#include "vvexport.h"

class VIRVOEXPORT vvvffile
{
  protected:
    unsigned char   FILEREAD;
    int             number_of_data_arrays, proceeded, checksum, headersize, iconsize;
    int             number_of_rgba_channels, number_of_channels, number_of_tf_channels;
    int             voxel_number_x, voxel_number_y, voxel_number_z, bytes_per_voxel, version_number;
    int             tailsize, icontype, textlength, data_array_length, hotspot_x, hotspot_y, hotspot_z;
    int             channel_array_length, time_stamp, temp_array_number, frames_number;
    int             number_of_tf_points, *number_of_pins, number_of_pin_lists, pin_data_length;
    int             i, j, k;
    float           realsize_x, realsize_y, realsize_z;
    unsigned char   a1, a2, a3, a4;
    char*           fileName;
    char*           descr, *channel_description;
    char*           tempfile;
    unsigned char*  icondata, *channel, *tf_channel, *data_array, *pin_data;
    int             errnmb;

    FILE*           dz, *dt;

    // reads all values in file
    int             readFile(void);

    // needed from float2octals
    int             mbit(float* wert);

    // returns 4x8Bits describing one float-value (IEEE-754 Single Precision)
    void            float2octals(unsigned char* a1, unsigned char* a2, unsigned char* a3, unsigned char* a4, float value);

    // returns float-value computed of 4x8Bits (IEEE-754 Single Precision)
    float           octals2float(unsigned char a1, unsigned char a2, unsigned char a3, unsigned char a4);

    // returns 4x8Bits describing one integer-value
    void            int2octals(unsigned char* a1, unsigned char* a2, unsigned char* a3, unsigned char* a4, int value);

    // returns integer-value computed of 4x8Bits
    int             octals2int(unsigned char a1, unsigned char a2, unsigned char a3, unsigned char a4);
  public:
    vvvffile(char* fName);
    virtual ~vvvffile(void);

    // not new, not nice - but still functioning
    // resolving header-data from raw-data files
    int     readXYZunsupp(void);

    // resolving data from raw-data files
    int     readDATAunsupp(unsigned char* data);

    // resolving header-data from '.xb7' files
    void    getResolution(int* x, int* y, int* z, int* b);
    int     getDataSize(void);
    int     readHeaderNtBoxBox(void);

    // resolving data from '.xb7' files
    int     readDataxyzvxvyvzrnc(unsigned char* data, int c);

    // NEW functions ... old algorithms in new environment
    // compress data from daten to coded_data
    void    encodeData(unsigned char* daten, int code, int para, unsigned char* coded_data);

    // decompress data from daten to decoded_data
    void    decodeData(unsigned char* daten, int lengt, int code, unsigned char* decoded_data);

    // returning the number of RGBA-Channels in vvvffile
    int     getNumberOfRGBAChannels(void);

    // returning the value of number_of_data_arrays in vvvffile
    int     getNumberOfDataArrays(void);

    // setting values for length, width, height and bytes/voxel
    int     setArrayDimensions(int x, int y, int z, int b);

    // returning length, width, height and bytes/voxel
    int     getArrayDimensions(int* x, int* y, int* z, int* b);

    // setting hotspot-coordinates
    int     setHotspot(int hotx, int hoty, int hotz);

    // returning hotspot-coordinates
    int     getHotspot(int* hotx, int* hoty, int* hotz);

    // setting real dimensions of a voxel
    int     setVoxelsize(float voxx, float voxy, float voxz);

    // returning real dimensions of a voxel
    int     getVoxelsize(float* voxx, float* voxy, float* voxz);

    // setting description of vf-File
    int     setFileDescription(char* text);

    // getting vf-File description
    char*   getFileDescription(void);

    // set the type of the icon and the data
    int     setIcon(int type, unsigned char* data);

    // get the type and length of the icon, returns pointer on icon data
    // DONT USE      unsigned char *getIcon( int *type, int *iconlength );
    // return the size of the icondata array
    int     getIconSize(void);

    // fill icondata[] and return type of icon
    int     getIcon(unsigned char* icondata);

    // create tempfile to combine data arrays
    int     setDataFile(void);

    // append a data array to tempfile
    int     addToDataFile(int data_array_length, unsigned char coding, unsigned char* data);

    // delete tempfile
    int     delDataFile(void);

    // best type = shortest result
    int     findBestTypeOfCoding(unsigned char* data, int* coding_parameter, int* best_length);

    // write complete vvvffile
    int     writeFile(void);

    // initialize RGBA-array
    int     setRGBAChannels(void);

    // add a RGBA-Channel to array
    int     addRGBAChannel(unsigned char* r_channel, unsigned char* g_channel, unsigned char* b_channel, unsigned char* a_channel, char* description);

    // opposite to setRGBAChannels
    int     delRGBAChannels(void);

    // retrieve the number'th RGBA-Channel from file
    int     getRGBAChannel(int number, unsigned char*  r_channel, unsigned char*  g_channel, unsigned char*  b_channel, unsigned char*  a_channel,
      char*  channel_description);

    // clear and prepare pin array
    int     initPins(void);

    // add a pin to array
    int     addPin(int list_number, int type, float value1, float value2, float value3, float xPos, char* description);

    // get pin data
    int     getPin(int list_number, int number, int* type, float* value1, float* value2, float* value3, float* xPos, int* descr_length);

    // get pin description
    int     getPinDescription(char* description);

    //
    int     initTFChannel(void);

    //
    int     addTFPoint(int position, int value);

    //
    int     delTFPoint(int position);

    //
    int     getTFPoints(int number, unsigned char* tf_channel, char* channel_description);

    //
    int     delTFChannel(void);

    //
    int     saveTFPoints(unsigned char* tf_channel, char* description, int type);

    //
    int     initChannels(void);

    //
    int     delChannels(void);

    // read data-array from file, 0 means all arrays
    int     readDataArray(unsigned char ** data_array, int array_number);

    // just in case you want to free up memory without calling destructor of class
    int     giveUpDataArray(void);

    // resolve needed space for a data array, 0 means for all arrays
    int     getDataArraySize(int number);

    // TO BE ELIMINATED IN THE FUTURE
    void    setValues(int vers_numb, int head_size, int temp_numb, char* temp_file);
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
