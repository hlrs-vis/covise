/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Switch>

using namespace std;
using namespace osg;

class FrameVisitor : public osg::NodeVisitor
{
private:
    bool _highdetail;
    bool _fade;

public:
    FrameVisitor();
    void setHighDetailOn(bool);
    void setFadeOn(bool);
    virtual void apply(osg::Switch &);
};
