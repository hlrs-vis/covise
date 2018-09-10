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

#ifndef VV_DICOM_H
#define VV_DICOM_H

#include "vvexport.h"
#include "vvinttypes.h"

/** Dicom properties.
  This is a storage class for value read from a DICOM file.
*/
class VIRVO_FILEIOEXPORT vvDicomProperties
{
  public:
    uchar* raw;                                   // data value array
    int    width, height;                         // width and height
    int    bpp;                                   // bytes per pixel
    int    chan;                                  // number of channels
    int    sequence;                              // sequence number
    int    image;                                 // image number
    int    bitsStored;                            // bits stored
    int    highBit;                               // most significant bit index
    bool   isSigned;                              // true = data is signed
    bool   littleEndian;                          // true = data is little endian
    float  dist[3];                               // sample distance (x,y,z)
    float  slicePos;                              // slice position
    float  sliceThickness;                        // slice thickness [mm]

    vvDicomProperties();
    void print();
};

/** dicom2pgm<P>
  Copyright (C) 1995 George J. Grevera<P>

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation.<P>

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.<P>

You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the Free
Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.<P>

Click <a href="http://www.rad.upenn.edu/~grevera/LICENSE">here</a>
for a copy of the license.<P>
<PRE>
Date  : 12/5/95
Revision history:
10/12/98 - added 18: 15,1400,1401,5101,6000
- also, sick of seeing warning about getopt being undefined
on some versions of unix so i #include <unistd.h>
7/13/98 - added 18: 93,94,1040,1062,1315.
5/10/98 - added support to output a 16-bit version of raw pgm files.
this also includes a function to read in the above data.
use the -f -r -o options to enable this.
3/12/98 - implemented speed up which only calls fprintf(textfp when
really desired (i.e. the "say" #define below)
3/4/98 - added explicit VR support (in addition to implicit)
- changed the default (to speed things up) so only -v (verbose)
will cause the contents of the dicom header to be output as
text
2/26/98 - added my own version of unix getopt() as it's missing in
windows
- noticed that win95/nt requires files to be open "rb" and
"wb" and not simply "r" and "w"
1/2/98 - corrected usage help text for specifying contrast
modification on the command line
- 'no flip' specified on the command should override the
default of 'flip'
- made dicom header contrast info public (useful for
callable version)
12/5/97 - replaced pow(2,i) with 1<<i as they are equivalent in
this context and the latter is faster than the former
- SEEK_SET isn't defined under SunOS (but is defined under
Solaris) so define it if necessary
11/12/97 - added function read_dicom_file to allow this code to be
called as a library function rather than as a standalone
program
10/ 9/97 - added support for MONOCHROME1 & MONOCHROME2 in
photometric interpretation (28,4)
9/ 3/97 - corrected color table allocation
8/25/97 - corrected pixel representation interpretation
8/13/97 - incorporated tiff output support as a command line option
</PRE>
@author George J. Grevera
@author Jurgen Schulze
*/
class VIRVO_FILEIOEXPORT vvDicom
{
  private:

    enum DicomTypes                               // DICOM data types

    {

      VV_UNKNOWN, VV_STRING, VV_FLOAT

    };

    vvDicomProperties* info;                      ///< DICOM properties
    int explicitVR;                               ///< explicit value representation flag (default is implicit)
    int photometric_specified;                    ///< support for photometric interpretation:
    ///<  0 = not specified
    ///<  1 = MONOCHROME1
    ///<  2 = MONOCHROME2
    ///<  4 = RGB
    int red_table_size;
    int green_table_size;
    int blue_table_size;
    ushort* red_table;
    ushort* green_table;
    ushort* blue_table;
    FILE* fp;                                     ///< DICOM input file

    void setEndian(void);
    int  read16(FILE* fp);
    int  read32(FILE* fp);
    void flip16bitData (uchar*, int);
    void flip16bitData (uchar*, int, int);
    void readRawPGM16File(char*, short**, int*, int*);
    void handlePhotometricInterpretation();
    void readDicomData(FILE*);

  public:
    int     window_center_specified, window_width_specified;
    double  window_center, window_width;
    int     slope_specified, intercept_specified;
    double  slope, intercept;

    vvDicom(vvDicomProperties*);
    bool readDicomFile(char*);
};
#endif

// EOF
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
