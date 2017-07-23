/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_WINDOW_H
#define VR_WINDOW_H

/*! \file
 \brief  window class for opening OpenGL contexts

 \author Daniela Rainer
 \author Uwe Woessner <woessner@hlrs.de> (osgProducer port)
 \author Martin Aumueller <aumueller@uni-koeln.de> (osgViewer port)
 \author (C) 1996
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   29.08.1997
 \date   09.07.1998 (Performer c++ interfce)
 \date   2004 (osgProducer)
 \date   2007 (osgViewer)
 */

#include "EventReceiver.h"

#include <util/coExport.h>
namespace opencover
{
class COVEREXPORT VRWindow
{

private:
    static VRWindow *s_instance;

    int *origVSize, *origHSize;

    bool createWin(int i);

    bool _firstTimeEmbedded;
    EventReceiver *_eventReceiver;

public:
    VRWindow();

    ~VRWindow();

    bool config();

    void lockPipes();
    void update();

    void setOrigVSize(int win, int size)
    {
        origVSize[win] = size;
    };
    void setOrigHSize(int win, int size)
    {
        origHSize[win] = size;
    };

    static VRWindow *instance();
};
}
#endif
