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

#ifndef VV_MOVIE_H
#define VV_MOVIE_H

// Virvo:
#include "vvtokenizer.h"
#include "vvsllist.h"

// Local:
#include "vvcanvas.h"

//============================================================================
// Class Definitions
//============================================================================

namespace vox 
{

class vvMovieStep;

/**
The Virvo shell allows the user to generate movies with changing camera
positions by loading ASCII movie script files (.vms).

The following movie script commands are recognized:

<PRE>
trans AXIS DIST
  Translates the data set by DIST in the AXIS axis.
  AXIS can be x, y, or z.

rot AXIS ANGLE
  Rotates the data set by ANGLE degrees about the AXIS axis.
  AXIS can be x, y, or z.

scale FACTOR
  Scales the data set by a factor of FACTOR. A value of 1.0 means no scaling.
  Values greater than 1.0 enlarge the data set, values smaller than 1.0
  make it smaller.

timestep INDEX
  Display time step number INDEX in a volume animation.
  The first time step has index 0.

nextstep
  Advance to the next time step.
  Will jump to the first step when the end is reached.

prevstep
  Go back to the previous time step.
  Will jump to the last step when the beginning is reached.

setpeak POS WIDTH
  Set a peak pin with WIDTH width [0..1] to POS [0..1].
  This will overwrite all previously defined alpha pins.

movepeak DISTANCE
  Move alpha peak by DISTANCE. The total alpha range has extension 1.0,
  so a DISTANCE value of 0.1 would move the peak by 1/10th of the
  value range to the right.

setquality QUALITY
  Set rendering quality. 0 is worst, 1 is default, higher is better

changequality RELATIVE_QUALITY
  Changes the quality setting by a relative value. Quality value cannot
  get smaller than zero.

setclip X Y Z POS
  Define and enable a clipping plane. Use X,Y,Z,POS=0 to disable.

moveclip DX DY DZ DPOS
  Move clipping plane relative to current position.

setclipparam SINGLE OPAQUE PERIMETER
  Set clipping plane parameters: 
    SINGLE: 1=single slice, 0=cutting plane
    OPAQUE: 1=if single slice then make opaque, 0=use transfer function settings for slice
    PERIMETER: 1=show clipping plane perimeter, 0=don't show perimeter

show
  Displays the data set using the current settings.

Here is an example movie script file:

scale 1.2       # scale object by factor 1.2
rot x 20        # rotate 20 degrees about the x axis
rot y 25        # rotate 25 degrees about y axis
timestep 0      # switch to first time step
show            # display dataset
timestep 1      # switch to second time step
show            # display dataset
repeat 10       # repeat the following 10 times
rot z 2         # rotate 2 degrees about z axis
rot x 1         # rotate 1 degree about x axis
show            # display dataset
endrep          # terminate repeat loop
rot z 10        # rotate 10 degrees about z axis
show            # display dataset
setpeak 0.0 0.1 # define peak at the lowest scalar value with width 0.1
show            # display dataset
repeat 5        # repeat the following 5 times
movepeak 0.1    # move peak to the right by 1/10th of the scalar value range
show            # display dataset
endrep          # terminate repeat loop

@author Jurgen P. Schulze (jschulze@ucsd.edu)
*/
class vvMovie
{
  protected:
    char* scriptName;                             ///< Filename of movie script, "" if undefined
    vvSLList<vvMovieStep*>* steps;                ///< pointer to list of movie steps, NULL if no movie present
    vox::vvCanvas* _canvas;
    size_t _currentStep;

  public:
    enum TransformType                            /// object modification types
    {
      NONE = 0, TRANS, ROT, SCALE, TIMESTEP, NEXTSTEP, PREVSTEP, MOVEPEAK, SETPEAK, 
      SETCLIP, MOVECLIP, SETCLIPPARAM, SHOW, SETQUALITY, CHANGEQUALITY
    };
    enum ErrorType                                /// Error Codes
    {
      VV_OK = 0,                                  ///< no error
      VV_FILE_ERROR,                              ///< file IO error
      VV_EOF,                                     ///< end of file
      VV_INVALID_PARAM,                           ///< invalid parameter
      VV_END_REPEAT                               ///< close repeat loop
    };

    vvMovie(vox::vvCanvas*);
    virtual ~vvMovie();
    ErrorType load(const char*);
    ErrorType parseCommand(vvTokenizer*, vvSLList<vvMovieStep*>*);
    const char* getScriptName();
    int   getNumScriptSteps();
    bool  setStep(size_t);
    size_t   getStep();
    bool  write(int, int, const char*);
};

/** One time step in a volume animation.
  @author Juergen Schulze-Doebold (schulze@hlrs.de)
  @see vvMovie
*/
class vvMovieStep
{
  public:
    enum {MAX_NUM_PARAM = 4};                     /// maximum number of parameters
    vvMovie::TransformType  transform;            ///< transformation type
    float                   param[MAX_NUM_PARAM]; ///< transformation parameters

    vvMovieStep()
    {
      transform = vvMovie::NONE;
      param[0] = param[1] = 0.0f;
    };
};

}
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
