/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PICKSPHERE_PLUGIN_H
#define PICKSPHERE_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>

namespace opencover
{
class BoxSelection;

namespace ui
{
class Button;
}
}

class VolumeOfInterestInteractor;

using namespace opencover;

struct MatrixState
{
    osg::Matrix matrix;
    float scaleFactor;
};

class VolumeOfInterestPlugin : public coVRPlugin, public ui::Owner
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
    ui::Button *m_useVolumeOfInterestCheckbox;
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
};
#endif
