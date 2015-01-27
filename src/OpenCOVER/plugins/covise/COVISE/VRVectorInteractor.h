/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRVECTOR_H
#define VRVECTOR_H

/*! \file
 \brief  3D interface for specifying vector values

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 1998
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   28.09.1998
 */

#include <util/common.h>
#include <util/DLinkList.h>

namespace osg
{
class MatrixTransform;
class Node;
}
namespace opencover
{
class buttonSpecCell;
class RenderObject;

// A VectorInteractor Attribute has the following form
// VECTOR%d %cmodule \n instance \n host \n parameterName \n x \n y \n z \n parameterName2 \n x \n y \n z \n scaleFactor\n)
//   number Type (M (Menu) , V(VR) or v(VR line not visible))
//																			or \n none \n

// class definitions
class VectorInteractor
{
public:
    static VectorInteractor *vector;
    static int menue;

    osg::MatrixTransform *transform; // transform of Sphere
    osg::Node *node; // Geometry node, this slider belongs to

    // return true, if ths attribute os from the same module/parameter
    int isVectorInteractor(const char *attrib);
    void updateMenu();
    void updateParameter();
    void update(buttonSpecCell *spec);
    void addMenue();
    char getType()
    {
        return (feedback_information[0]);
    };
    void updateValue(float x, float y, float z);
    float getMinDist(float x, float y, float z);
    static void menuCallback(void *sider, buttonSpecCell *spec);
    static osg::Node *getArrow();
    VectorInteractor(const char *attrib, const char *sattrib, osg::Node *n);
    ~VectorInteractor();

private:
    int move;
    int rotOnly;
    void updatePosition();
    char *feedback_information;
    char *sattrib;
    char *subMenu;
    char *moduleName;
    char *parameterName;
    char *parameterName2;
    float x1, y1, z1;
    float x2, y2, z2;
    float scaleFactor;
};
class VectorInteractorList : public covise::DLinkList<VectorInteractor *>
{
public:
    /// add all VectorInteractors defined in this Do to the menue
    /// if they are not jet there
    /// otherwise update the node field
    //void add( coDistributedObject *dobj, osg::Node *n);
    void add(RenderObject *dobj, osg::Node *n);
    VectorInteractor *find(osg::Node *geode);
    void removeAll(osg::Node *geode);
    VectorInteractor *find(float x, float y, float z);
    VectorInteractor *find(const char *attrib);
};

// global stuff
extern VectorInteractorList vectorList;
}
// done
#endif
