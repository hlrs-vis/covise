// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
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

#ifndef VVCONV_H
#define VVCONV_H

#include <virvo/math/forward.h>
#include <virvo/vvvoldesc.h>

/** Command line volume file format converter.
  Usage: Type 'vconv' to get a list of command line parameters, or
  view the main() function in vvconv.cpp.<BR>
  This program supports the following macro definitions:
  <DL>
    <DT>VV_DICOM_SUPPORT</DT>
    <DD>If defined, the Papyrus library is used and DICOM files can be read.</DD>
  </DL>
  @author Juergen P. Schulze (jschulze@ucsd.de)
*/
class vvConv
{
  private:
    vvVolDesc* vd;      ///< description of processed volume
    char* srcFile;      ///< source file name
    char* dstFile;      ///< destination file name
    int   entry;        ///< number of entry to process (only when reading DICOM files with GDCM)
    int   files;        ///< number of files for slice reading or transient data, 1 is default
    int   increment;    ///< file name increment (default: 1)
    bool  showHelp;     ///< display help information
    bool  fileInfo;     ///< display file information
    bool  overwrite;    ///< true = overwrite destination file if it exists
    int   bpchan;       ///< bytes per channel, -1 for don't change
    bool  sphere;       ///< true = convert flat data to sphere data
    int   outer;        ///< diameter of outer sphere [voxels]
    int   inner;        ///< diameter of inner sphere [voxels]
    bool  heightField;  ///< true = calculate height field from slice
    int   hfHeight;     ///< height field height [voxels]
    int   hfMode;       ///< height field computation mode [0|1]
    bool  showbounds;   ///< true = show boundary of non-zero data
    bool  crop;         ///< true = crop data
    int   cropPos[3];   ///< crop position (x,y,z) [voxels]
    int   cropSize[3];  ///< crop size (width, height, slices) [voxels]
    bool  croptime;     ///< true = crop time steps
    int   cropSteps[2]; ///< first time step and number of time steps to extract
    bool  croptodata;   ///< true = crop volume to non-zero data
    bool  resize;       ///< true = resize volume
    int   newSize[3];   ///< new volume size (width, height, slices) [voxels]
    float resizeFactor; ///< alternatively to new sizes a factor is accepted
    bool  replace;      ///< data replacement mode
    int   replaceOld[3], replaceNew[3]; ///< data values to be replaced
    bool  setDist;      ///< true = set voxel distance
    float newDist[3];   ///< new voxel distance [mm]
    bool  setRange;     ///< true = set physical scalar value range
    float newRange[2];  ///< new realMin and realMax values
    bool  flip;         ///< true = flip volume
    virvo::cartesian_axis< 3 > flipAxis; ///< flip axis
    bool  rotate;       ///< true = rotate volume
    int   rotDir;       ///< rotation direction: -1=negative, +1=positive, 0=invalid parameter
    virvo::cartesian_axis< 3 > rotAxis;  ///< rotation axis
    bool  swap;         ///< true = byte swapping on
    bool  shift;        ///< true = shift volume
    int   shiftDist[3]; ///< shift distance [voxels]
    bool  bitshift;     ///< true = bit shift voxel values
    int   bshiftDist;   ///< bit shift distance [bits]
    bool  importTF;     ///< true = import transfer functions from a file
    bool  removeTF;     ///< true = remove all transfer functions
    char* importFile;   ///< name of file providing the transfer functions
    bool  drawLine;     ///< true = draw a 3D line into the dataset
    int   lineStart[3]; ///< starting point of line
    int   lineEnd[3];   ///< end point of line
    int   lineColor;    ///< line color value
    bool  drawBox;      ///< true = draw a 3D box into the dataset
    int   boxStart[3];  ///< starting point of box
    int   boxEnd[3];    ///< end point of box
    int   boxColor;     ///< box color value
    bool  loadRaw;      ///< true = load file in raw mode
    int   rawBPC;       ///< byte per channel in raw file
    int   rawCh;        ///< number of channels in raw file
    int   rawWidth, rawHeight, rawSlices; ///< volume size in raw file
    ssize_t rawSkip;    ///< number of bytes to skip in a raw file (to ignore header)
    bool  signedData;   ///< true = make data unsigned
    vvVolDesc::InterpolationType ipt;     ///< interpolation type to use for resampling
    bool  statistics;   ///< true = print statistics about volume data
    bool  fillRange;    ///< true = expand data range to use all values from 0 to maximum
    bool  dicomRename;  ///< true = rename DICOM files
    bool  leicaRename;  ///< true = rename Leica files
    bool  compression;  ///< true = compress data if allowed by file format
    float animTime;     ///< time that each animation frame is to be displayed [seconds], 0=no change
    bool  deinterlace;  ///< true = deinterlace slices
    bool  zoomData;     ///< true = zoom data range
    int   zoomRange[3]; ///< channel, bottom and top limits of zoom range [scalar values]
    bool  loadXB7;      ///< true=force XB7 format
    int   xb7Size;      ///< volume edge length [voxels]
    int   xb7Param;     ///< index of parameter to use as scalar value
    bool  xb7Global;    ///< true=use global min/max for scaling, false=use time step local min/max for scaling
    bool  loadCPT;      ///< true=force checkpoint format
    int   cptSize;      ///< volume edge length [voxels]
    int   cptParam;     ///< index of parameter to use as scalar value
    bool  cptGlobal;    ///< true=use global min/max for scaling, false=use time step local min/max for scaling
    bool  setPos;       ///< true = set position in object space
    float posObj[3];    ///< position in object space
    bool  histogram;    ///< true = print histogram as text
    int   histType;     ///< 0=ASCII, 1=PGM image
    bool  sign;         ///< true = toggle sign
    bool  blend;        ///< true = do blending
    char* blendFile;    ///< file to blend
    int   blendType;    ///< blending method
    bool  mask;         ///< true = apply binary mask
    char* maskFile;     ///< file containing the binary mask
    int   channels;     ///< set number of channels (-1 if not used)
    bool  setIcon;      ///< true = set icon to image from file
    char* iconFile;     ///< icon image file
    bool  makeIcon;     ///< true = make icon from slices
    int   makeIconSize; ///< new icon size [pixels]
    bool  getIcon;      ///< true = extract icon to file
    bool  swapChannels; ///< true = swap channels
    int   swapChan[2];  ///< channel IDs to swap (0=first channel)
    bool  extractChannel; ///< true = extract 4th channel from RGB
    float extract[3];   ///< channel extraction parameters
    int   makeVolume;   ///< create volume algorithmically; default: -1
    int   makeVolumeSize[3]; ///< size of volume [voxels]
    char* addFile;      ///< file for addchannel
    bool  addChannel;   ///< true = add a channel from file
    bool  autoRealRange;
    bool  invertVoxelOrder; ///< true = invert innermost and outermost voxel loops
    int lineAverage;
    int sections;
    int pinhole;
    int height;
    int width;        // The data being used to rename the leica files
    int* laser;
    int* intensity;
    int* gain;
    int* offset;
    int lasercount;
    float thickness;
    int starttime;
    int endtime;
    int chan[3];
    int mergeType;    ///< 0=none specified, 1=volume, 2=animation
    
    bool readVolumeData();
    bool writeVolumeData();
    void displayHelpInfo();
    bool parseCommandLine(int, char**);
    void modifyInputFile(vvVolDesc*);
    void modifyOutputFile(vvVolDesc*);
    int  renameDicomFiles();

  public:
    vvConv();
    ~vvConv();
    int run(int, char**);
};

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
