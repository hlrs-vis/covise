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

#ifdef VV_USE_ARTOOLKIT

#ifndef VV_ARTRACKING_H
#define VV_ARTRACKING_H

// ARToolkit:
#include <AR/gsub.h>
#ifdef _WIN32
  #include <AR/ARFrameGrabber.h>
#endif

class vvMatrix;

namespace vox
{

/** This class allows using an ARToolkit marker as a 6DOF input device.
  It requires header files and libraries from ARToolkit and MS DirectShow.
  References:
  - ARToolkit: http://www.hitl.washington.edu/artoolkit/

  TODO: add ARTag functions, see www.artag.net
  @author Juergen Schulze
*/
class vvARTracking
{
  private:
    ARFrameGrabber _camera;
    char*   _cparamPath;
    char*   _patternPath;
    double  _patternXF[3][4];
    int     _patternID;
    int     _patternWidth;
    bool    _showPropertiesDialog;
    ARUint8* _dataPtr;
    int     _videoSize[2];

  public:
    vvARTracking();
    virtual ~vvARTracking();
    int init();
    void track();
    void showPropertiesDialog(bool);
    void setCameraParamPath(const char*);
    void setPatternPath(const char*);
    void setPatternWidth(int);
    vvMatrix getMarkerMatrix();
    unsigned char* getVideoImage();
};

}
#endif

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
