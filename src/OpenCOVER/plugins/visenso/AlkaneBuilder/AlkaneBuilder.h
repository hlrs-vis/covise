/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ATOM_BUILDER_H
#define _ATOM_BUILDER_H

#define MAX_CARBONS 9
#define MAX_HYDROGENS 20

#define CARBON_POS_X 300
#define CARBON_POS_Y 50
#define CARBON_POS_Z 50
#define HYDROGEN_POS_X 300
#define HYDROGEN_POS_Y -50
#define HYDROGEN_POS_Z -50
#define DESCRIPTION_POS_X -400
#define DESCRIPTION_POS_Y 0
#define DESCRIPTION_POS_Z 200

#include "AlkaneDatabase.h"

#include <osg/Vec3>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <vector>
#include <osgText/Text>
#include <PluginUtil/coPlane.h>
class Atom;
class Carbon;
namespace opencover
{
class coHud;
}

class AlkaneBuilder
{
public:
    AlkaneBuilder();
    ~AlkaneBuilder();
    void setAlkane(Alkane a);
    void update();
    void show(bool value);
    void reset();
    void setModeBuild(bool mode);
    void enableIntersection(bool);
    void showPlane(bool);
    void showInstructionText(bool);
    void showStatusText(bool);
    void setLinear(bool);
    void showErrorPanel();
    bool getStatus()
    {
        return forwardOk_;
    };

private:
    Alkane *currentAlkane_;
    std::vector<Atom *> atoms_;
    std::vector<Carbon *> carbons_;
    osg::ref_ptr<osg::Group> group_;
    osg::ref_ptr<osg::MatrixTransform> planeTransformNode_;
    osg::ref_ptr<osg::Geode> planeNode_;
    osg::ref_ptr<osg::Geode> anweisungGeode_, statusGeode_;
    float size_;
    osgText::Text *anweisungText_, *statusText_;
    opencover::coPlane *plane_;
    bool mode_;

    void check(); //check if alkane is configured correctly
    void makeDescription(std::string name, std::string formula, float fontSize, osg::Vec3 pos);
    void updateDescription(std::string name, std::string formula);

    //bool isCH4(Atom*);
    //bool isC2H6(Atom*);
    //bool isC3H8(Atom*);
    //Atom *isCH3(Atom*);
    //Atom *isC2H2(Atom*, Atom*);

    void buildAlkaneStart(Carbon *c, Atom *h0, Atom *h1, Atom *h2);
    void buildAlkaneMiddle13(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1);
    void buildAlkaneMiddle20(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1);
    void buildAlkaneEnd13(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1, Atom *h2);
    void buildAlkaneEnd20(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1, Atom *h2);

    void buildAlkane(int nc, int nh, bool linear);
    void createPlane();

    opencover::coHud *hud_; // hud for message
    float hudTime_;
    bool forwardOk_;
    bool linear_;
};

#endif
