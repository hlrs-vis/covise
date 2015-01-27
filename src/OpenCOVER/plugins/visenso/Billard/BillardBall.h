/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BILLARD_BALL_H
#define _BILLARD_BALL_H
namespace opencover
{
class coVR3DTransRotInteractor;
}
using namespace opencover;

#include <string>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <osg/Matrix>

class BillardBall : public coVR3DTransRotInteractor
{
private:
    std::string geoFileName_;

protected:
    void createGeometry();

public:
    // position and normal in object coordinates
    // size in world coordinates (mm)
    BillardBall(osg::Matrix initialMat, float size, std::string geofilename);
    virtual ~BillardBall();
    virtual void keepSize();
    osg::Node *getNode()
    {
        return moveTransform.get();
    };
};

#endif
