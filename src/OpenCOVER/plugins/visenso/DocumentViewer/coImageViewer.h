/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_IMAGE_VIEWER_H
#define _CO_IMAGE_VIEWER_H

#include <OpenVRUI/coMenu.h>

#include <vector>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coMovableBackgroundMenuItem;
class coSliderMenuItem;
class coButtonMenuItem;
}

// image viewer is
// submenu-item in COVER main menu
// menu
//   list of movable background items, but only one in menu
//   zoom slider item
//   page selection slider item

class coImageViewer : public vrui::coMenuListener
{
public:
    coImageViewer(const char *documentName, const char *imagePath, float vsize, float ascpect_ratio); //vsize is vertical size (height) in mm
    virtual ~coImageViewer();

    // add a image
    bool addImage(const char *imageName);

    // update slider with current image scale
    void preFrame();

    // menu event of slider
    virtual void menuEvent(vrui::coMenuItem *);

    // get document name
    const char *getName();

    // open/close document from remote
    void setVisible(bool visible);

    // show page number...
    void setPageNo(int no);

    // get number of pages
    int getNumPages();

    // set size
    void setSize(int pageNo, float hsize, float vsize);

    // get size
    float getVSize(int pageNo);

    // get aspect
    float getAspect(int pageNo);

    // show page number...
    void setScale(float s);

    // position the documenz
    void setPosition(float x, float y, float z);

    // name of the obj
    void setObjName(const char *objName);

private:
    vrui::coSubMenuItem *pinboardButton_; // button in main menu
    vrui::coRowMenu *imageMenu_; // submenu
    // image item in submenu
    std::vector<vrui::coMovableBackgroundMenuItem *> imageItemList_;
    int currentPageIndex_;
    vrui::coSliderMenuItem *zoomSlider_;
    vrui::coSliderMenuItem *pageSlider_;
    vrui::coButtonMenuItem *resetZoomButton_;
    char *documentName_;
    float aspect_; // aspect of first image
    float vsize_; //  vertical size of firts image
    bool documentInMenu_;
    const char *objName_;

    bool pageSliderInMenu;
};
#endif
