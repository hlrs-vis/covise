/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PICKSPHERE_PLUGIN_H
#define PICKSPHERE_PLUGIN_H

#include <OpenVRUI/coMenuItem.h>
#include <cover/coVRPlugin.h>

namespace vrui
{
class coCheckboxMenuItem;
}
namespace opencover
{
class BoxSelection;
}

class VolumeOfInterestInteractor;

using namespace vrui;
using namespace opencover;

struct MatrixState
{
    osg::Matrix matrix;
    float scaleFactor;
};

class VolumeOfInterestPlugin : public coVRPlugin, public coMenuListener
{
public:
    VolumeOfInterestPlugin();
    virtual ~VolumeOfInterestPlugin();
    bool init();
    static VolumeOfInterestPlugin *instance();

    void preFrame();
    static void defineVolumeCallback();
    void defineVolume();
    static void setToPreviousStateCallback();
    void setToPreviousState();

private:
    std::vector<MatrixState> m_stateHistory;
    coCheckboxMenuItem *m_useVolumeOfInterestCheckbox;
    VolumeOfInterestInteractor *m_volumeOfInterestInteractor;

    osg::Matrix m_originalMatrix, m_destinationMatrix;
    float m_originalScaleFactor, m_destinationScaleFactor;
    int m_count;
    bool m_reset, m_firstTime, m_volumeChanged;
    osg::Vec3 m_min, m_max;

    static BoxSelection *s_boxSelection;

    void createMenuEntry();
    void deleteMenuEntry();
    void setCurrentInterpolationState(float alpha);

    void menuEvent(coMenuItem *);
};

#endif
