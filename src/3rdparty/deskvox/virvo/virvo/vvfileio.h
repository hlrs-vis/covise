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

#ifndef VV_FILEIO_H
#define VV_FILEIO_H

#include <cassert>
#include "vvexport.h"
#include "vvvoldesc.h"

/** File load and save routines for volume data.
  The following file formats are supported:<BR>
  RVF, XVF, VF, AVF, 3D TIFF, Visible Human, raw, RGB, TGA, PGM, PPM<BR>
  When a 2D image file is loaded, the loader looks for a numbered sequence
  automatically, for instance: file001.rvf, file002.rvf, ...
  @author Juergen Schulze (schulze@hlrs.de)
*/

struct vvTifData;

class VIRVO_FILEIOEXPORT vvFileIO
{
  public:
    class ParticleTimestep
    {
      public:
        size_t numParticles;                         // number of particles
        float* pos[3];                            // x/y/z positions
        float* val;                               // values
        float min,max;                            // minimum and maximum scalar values
        ParticleTimestep(size_t np)
        {
          numParticles = np;
          min = max = 0;
          for(int i=0; i<3; ++i)
          {
            pos[i] = new float[numParticles];
            assert(pos[i]);
          }
          val = new float[numParticles];
          assert(val);
        }
        ~ParticleTimestep()
        {
          for(int i=0; i<3; ++i) delete[] pos[i];
          delete[] val;
        }
    };
    enum ErrorType                                /// Error Codes
    {
      OK,                                         ///< no error
      PARAM_ERROR,                                ///< parameter error
      FILE_ERROR,                                 ///< file IO error
      FILE_EXISTS,                                ///< file exists error
      FILE_NOT_FOUND,                             ///< file not found error
      DATA_ERROR,                                 ///< data format error
      FORMAT_ERROR,                               ///< file format error (e.g. no valid TIF file)
      VD_ERROR                                    ///< volume descriptor (vvVolDesc) error
    };
    enum LoadType                                 /// Load options
    {
      ALL_DATA = 0xFFFF,                          ///< load all data
      HEADER   = 0x0001,                          ///< load header
      ICON     = 0x0002,                          ///< load icon
      RAW_DATA = 0x0004,                          ///< load volume raw data
      TRANSFER = 0x0008                           ///< load transfer functions
    };

    vvFileIO();
    ErrorType saveVolumeData(vvVolDesc *, bool, LoadType sec = ALL_DATA);
    ErrorType loadVolumeData(vvVolDesc*, LoadType sec = ALL_DATA, bool addFrame=false);
    ErrorType loadDicomFile(vvVolDesc*, int* = NULL, int* = NULL, float* = NULL);
    ErrorType loadRawFile(vvVolDesc*, size_t, size_t, size_t, size_t, size_t, size_t);
    ErrorType loadXB7File(vvVolDesc*,int=128,int=8,bool=true);
    ErrorType loadCPTFile(vvVolDesc*,int=128,int=8,bool=true);
    ErrorType mergeFiles(vvVolDesc*, int, int, vvVolDesc::MergeType);
    void      setCompression(bool);
    ErrorType importTF(vvVolDesc*, const char*);

  protected:
    char _xvfID[10];                               ///< XVF file ID
    char _nrrdID[9];                               ///< nrrd file ID
    int  _sections;                                ///< bit coded list of file sections to load
    bool _compression;                             ///< true = compression on (default)

    void setDefaultValues(vvVolDesc*);
    int  readASCIIint(FILE*);
    bool parseLeicaFilename(const std::string, int32_t&, int32_t&, std::string&);
    bool changeLeicaFilename(std::string&, int32_t, int32_t);
    void makeLeicaFilename(const char*, int32_t, int32_t, char*);
    ErrorType loadWLFile(vvVolDesc*);
    ErrorType loadASCFile(vvVolDesc*);
    ErrorType saveRVFFile(const vvVolDesc*);
    ErrorType loadRVFFile(vvVolDesc*);
    ErrorType saveXVFFile(vvVolDesc*);
    ErrorType loadXVFFileOld(vvVolDesc*);
    ErrorType loadXVFFile(vvVolDesc*);
    ErrorType saveAVFFile(const vvVolDesc*);
    ErrorType loadAVFFile(vvVolDesc*);
    ErrorType loadVTKFile(vvVolDesc*);
    ErrorType loadTIFFile(vvVolDesc*, bool addFrame=false);
    ErrorType loadTIFSubFile(vvVolDesc*, FILE *fp, virvo::serialization::EndianType endian, long &nextIfdPos, vvTifData *tifData);
    ErrorType saveTIFSlices(const vvVolDesc*, bool);
    ErrorType loadRawFile(vvVolDesc*);
    ErrorType saveRawFile(const vvVolDesc*);
    ErrorType loadRGBFile(vvVolDesc*);
    ErrorType loadTGAFile(vvVolDesc*);
    ErrorType loadPXMRawImage(vvVolDesc*);
    ErrorType loadVHDAnatomicFile(vvVolDesc*);
    ErrorType loadVHDMRIFile(vvVolDesc*);
    ErrorType loadVHDCTFile(vvVolDesc*);
    ErrorType loadVMRFile(vvVolDesc*);
    ErrorType loadVTCFile(vvVolDesc*);
    ErrorType loadNiftiFile(vvVolDesc* vd);
    ErrorType saveNiftiFile(const vvVolDesc* vd);
    ErrorType loadNrrdFile(vvVolDesc*);
    ErrorType saveNrrdFile(const vvVolDesc*);
    ErrorType loadXIMGFile(vvVolDesc*);
    ErrorType loadVis04File(vvVolDesc*);
    ErrorType loadHDRFile(vvVolDesc*);
    ErrorType loadVOLBFile(vvVolDesc*);
    ErrorType loadDDSFile(vvVolDesc*);
    ErrorType loadGKentFile(vvVolDesc*);
    ErrorType loadSynthFile(vvVolDesc*);
    ErrorType savePXMSlices(const vvVolDesc*, bool);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
