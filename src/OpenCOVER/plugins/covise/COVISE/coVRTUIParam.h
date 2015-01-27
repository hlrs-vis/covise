/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRTUIPARAM_H
#define VRTUIPARAM_H

/*! \file
 \brief  make COVISE module parameters available via tablet user interface

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   28.04.2004
 */

// by Uwe Woessner
// 28.04.2004

// includes
#include <util/DLinkList.h>

#include <cover/coTabletUI.h>
namespace osg
{
class Node;
}

namespace opencover
{
class buttonSpecCell;
class RenderObject;
class coTUIElement;

#define TUI_FLOAT_SLIDER 0
#define TUI_INT_SLIDER 1
#define TUI_FLOAT 2
#define TUI_INT 3
#define TUI_BOOL 4

// A TUIParam Attribute has the following form
// TUI%d %cmodule \n instance \n host \n parameterName \n parent\n text \n xPos \n yPos \n floatSlider \n parameterName \n min \n max \n value)

// class definitions
class TUIParamParent : public coTUIListener
{
public:
    char *name;
    coTUIElement *element;

    TUIParamParent(const char *n);
    virtual ~TUIParamParent();
    virtual void tabletEvent(coTUIElement *);

private:
};

// class definitions
class TUIParam : public coTUIListener
{
public:
    int type;
    int xPos, yPos;
    float min, max, value, step;
    bool state;
    coTUIElement *element;
    TUIParamParent *parent;
    osg::Node *node;
    double lastTime;
    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);
    virtual void tabletEvent(coTUIElement *);
    int isTUIParam(const char *n);
    TUIParam(const char *attrib, const char *sattrib, osg::Node *n);
    virtual ~TUIParam();

private:
    char *feedback_information;
    char *moduleName;
    char *sattrib;
    char *parameterName;
    char *parameterText;
    void updateParameter(bool force);
    void exec();
};
class TUIParamParentList : public covise::DLinkList<TUIParamParent *>
{
public:
    TUIParamParent *find(const char *name);
};
class TUIParamList : public covise::DLinkList<TUIParam *>
{
public:
    /// add all TUIParams defined in this Do to the menue
    /// if they are not jet there
    /// otherwise update the node field
    //void add( coDistributedObject *dobj, osg::Node *n);
    void add(RenderObject *robj, osg::Node *n);
    TUIParam *find(osg::Node *geode);
    void removeAll(osg::Node *geode);
    TUIParam *find(const char *attrib);
};

// global stuff
extern TUIParamList tuiParamList;
extern TUIParamParentList tuiParamParentList;
}
// done
#endif
