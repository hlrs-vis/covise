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

#ifdef WIN32
  #include <windows.h>
#endif
#include <iostream>

// ARToolkit:
#include <AR/gsub.h>
#include <AR/video.h>

// Local:
#include "vvartracking.h"

using namespace vox;
using namespace std;

//----------------------------------------------------------------------------
/// Constructor
vvARTracking::vvARTracking()
{
  const char* cparamDefault  = "..\\camera_para.dat";
  const char* patternDefault = "..\\patt.hiro";
  _patternWidth = 80;
  _cparamPath  = new char[strlen(cparamDefault)  + 1];
  _patternPath = new char[strlen(patternDefault) + 1];
  strcpy(_cparamPath,  cparamDefault);
  strcpy(_patternPath, patternDefault);
  _showPropertiesDialog = false;
}

//----------------------------------------------------------------------------
/// Destructor
vvARTracking::~vvARTracking()
{
#ifdef _WIN32
	CoInitialize(NULL);
#endif
  //argCleanup();   // TODO: why is this crashing it?
#ifdef _WIN32
	CoUninitialize();
#endif
  delete[] _cparamPath;
  delete[] _patternPath;
}

//----------------------------------------------------------------------------
/** @param show true=display camera properties dialog in init() routine
*/
void vvARTracking::showPropertiesDialog(bool show)
{
  _showPropertiesDialog = show;
}

//----------------------------------------------------------------------------
void vvARTracking::setCameraParamPath(const char* newPath)
{
  delete[] _cparamPath;
  _cparamPath = new char[strlen(newPath) + 1];
  strcpy(_cparamPath, newPath);
}

//----------------------------------------------------------------------------
void vvARTracking::setPatternWidth(int newWidth)
{
  _patternWidth = newWidth;
}

//----------------------------------------------------------------------------
void vvARTracking::setPatternPath(const char* newPath)
{
  delete[] _patternPath;
  _patternPath = new char[strlen(newPath) + 1];
  strcpy(_patternPath, newPath);
}

//----------------------------------------------------------------------------
/** Initialize ARToolkit, especially camara parameters and
  marker pattern.
  @return 0 if ok, -1 if camera parameter load error, -2 if pattern load error
*/
int vvARTracking::init()
{
  const int XSIZE = 320;    // TODO: allow arbitrary video image size
  const int YSIZE = 240;
  ARParam cparam;
	ARParam wparam;

#ifdef WIN32
	CoInitialize(NULL);
#endif

  // Set the initial camera parameters:
  if(arParamLoad(_cparamPath, 1, &wparam) < 0) 
  {
    cerr << "ARToolkit: Camera parameter load error" << endl;
    return -1;
  }
  arParamChangeSize(&wparam, XSIZE, YSIZE, &cparam);
  arInitCparam(&cparam);
  cerr << "ARToolkit camera parameters:" << endl;
  arParamDisp(&cparam);
  
  if((_patternID = arLoadPatt(_patternPath)) < 0) 
  {
    cerr << "ARToolkit: Pattern load error" << endl;
    return -2;
  }

	// Start the video capture:
	_camera.Init(0);
  if (_showPropertiesDialog)
  {
    _camera.DisplayProperties();
  }

#ifdef WIN32
	CoUninitialize();
#endif

  // TODO: why is this not in the libraries?
//  if(arVideoInqSize(&_videoSize[0], &_videoSize[1]) >= 0)
  {
    cerr << "Video size: " << _videoSize[0] << " x " << _videoSize[1] << endl;
  }

  cerr << "ARToolkit tracking ready." << endl;
  return 0;
}

//----------------------------------------------------------------------------
void vvARTracking::track()
{
  const int THRESH = 100;
  static bool firstPass = true;
  ARMarkerInfo* marker_info;
  int marker_num;
  int j, k;
  double patternCenter[2] = {0.0, 0.0};

#ifdef WIN32
	CoInitialize(NULL);
#endif

  // Grab a video frame:
  _camera.GrabFrame();
  if((_dataPtr = (ARUint8*)_camera.GetBuffer()) == NULL) 
  {
    arUtilSleep(2); 
    return;
  }

  if (firstPass)
  {
    arUtilTimerReset();  
    firstPass = false;
  }

  // Detect the markers in the video frame:
  if(arDetectMarker(_dataPtr, THRESH, &marker_info, &marker_num) < 0 ) 
  {
    cerr << "arDetectMarker error" << endl; 
    argCleanup();
    exit(0);
  }

  // Check for object visibility:
  k = -1;
  for( j = 0; j < marker_num; j++ ) 
  {
    if(_patternID == marker_info[j].id) 
    {
      if( k == -1 ) k = j;
      else if( marker_info[k].cf < marker_info[j].cf ) k = j;
    }
  }

  if( k != -1 ) // has marker been found?
  {
    // Get the transformation between the marker and the real camera:
    arGetTransMat(&marker_info[k], patternCenter, _patternWidth, _patternXF);
  }
#ifdef WIN32
	CoUninitialize();
#endif	
}

//----------------------------------------------------------------------------
/** @return marker matrix in Virvo matrix format
*/
vvMatrix vvARTracking::getMarkerMatrix()
{
  double glPatternXF[16];
  argConvGlpara(_patternXF, glPatternXF);
  vvMatrix mat;
  mat.getGL(glPatternXF);
  return mat;
}

//----------------------------------------------------------------------------
unsigned char* vvARTracking::getVideoImage()
{
  return (unsigned char*)_dataPtr;
}

//----------------------------------------------------------------------------
#ifdef VV_STANDALONE
int main()
{
  vvARTracking* art = new vvARTracking();
  art->showPropertiesDialog(false);
  art->init();

  while(true)
  {
    art->track();
  }

  delete art;
  return 0;
}

#endif

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
