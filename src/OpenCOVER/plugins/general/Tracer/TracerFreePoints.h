/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACER_FREE_POINTS_H
#define _TRACER_FREE_POINTS_H

namespace opencover
{
class coVR3DTransInteractor;
}
namespace vrui
{
class coTrackerButtonInteraction;
}

#include <util/DLinkList.h>
class TracerFreePoints
{
private:
    opencover::coInteractor *_inter;
    std::vector<opencover::coVR3DTransInteractor *> _pointsList;
    vrui::coTrackerButtonInteraction *_directInteractor;

    float _interSize;

    int _numPoints;

    bool _newModule;
    bool showPickInteractor_;
    bool showDirectInteractor_;
    bool _waitForNewPoint;
    TracerPlugin *plugin;

public:
    // constructor
    TracerFreePoints(opencover::coInteractor *inter, TracerPlugin *p);

    // destructor
    ~TracerFreePoints();

    // update after module execute
    void update(opencover::coInteractor *inter);

    // set new flag
    void setNew();

    // direct interaction
    void preFrame();

    // show/hide interactors
    void showDirectInteractor();
    void showPickInteractor();
    void hideDirectInteractor();
    void hidePickInteractor();

    void setCaseTransform(osg::MatrixTransform *t);

private:
    TracerFreePoints();
};

#endif
