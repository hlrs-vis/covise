/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRcoVRMenuImage_H
#define VRcoVRMenuImage_H

/*! \file
 \brief  3D coVRMenuImage for specifying scalar values

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 1998
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   28.09.1998
 */

#include <util/DLinkList.h>

#include <osg/Node>

#include <util/coTypes.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>

namespace vrui
{
class coLabel;
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coTrackerButtonInteraction;
class coTexturedBackground;
class coPopupHandle;
class coRowContainer;
class coColoredBackground;
}

namespace opencover
{
class buttonSpecCell;
class RenderObject;

// class definitions
class coVRMenuImage
{
public:
    osg::ref_ptr<osg::Node> node; // Geometry node, this coVRMenuImage belongs to

    // return true, if ths attribute os from the same module/parameter
    std::string &getName()
    {
        return imageName;
    };
    std::string &getNodeName()
    {
        return nodeName;
    };

    coVRMenuImage(std::string name, osg::Node *n);
    ~coVRMenuImage();
    void updateImage(osg::Node *n);

    void show();
    void hide();
    void update();

    vrui::coCheckboxMenuItem *menuItem;

private:
    std::string imageName;
    std::string nodeName;
    vrui::coPopupHandle *popupHandle;
    vrui::coTexturedBackground *imageBackground;
};

class coVRMenuList : public covise::DLinkList<coVRMenuImage *>, public vrui::coMenuListener
{
public:
    static coVRMenuList *instance();

    bool add(opencover::RenderObject *robj, osg::Node *n); // returns true , if the node should be added to the scenegraph
    coVRMenuImage *find(osg::Node *geode);
    void removeAll(std::string nodeName);
    coVRMenuImage *find(std::string &name);
    void menuEvent(vrui::coMenuItem *menuItem);
    vrui::coSubMenuItem *pinboardEntry;
    vrui::coRowMenu *plotMenu;

    void update();

private:
    coVRMenuList();
    coVRMenuList(const coVRMenuList &);
    ~coVRMenuList();
};
}
// done
#endif
