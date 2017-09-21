/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DOCUMENT_VIEWER_PLUGIN_H
#define _DOCUMENT_VIEWER_PLUGIN_H

#include <string>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <OpenVRUI/coMenu.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
}

class coImageViewer;

using namespace vrui;
using namespace opencover;

class DocumentViewerPlugin : public coVRPlugin, coMenuListener
{
public:
    DocumentViewerPlugin();
    virtual ~DocumentViewerPlugin();
    bool init();

    void preFrame();
    void registerObjAtUi(std::string name);
    void addObject(const RenderObject *container, osg::Group *root, const RenderObject *, const RenderObject *, const RenderObject *, const RenderObject *);
    void removeObject(const char *objName, bool replace);
    void guiToRenderMsg(const char *msg);

    coMenuItem *getMenuButton(const std::string &buttonName);

    coSubMenuItem *coverMenuButton_;
    coRowMenu *documentsMenu_;
    coCheckboxMenuItem *toggleDocumentsButton_;

    // add a new document
    // - for each entry in covise.config
    //   if document is already in list, add image
    // - for each covise object with attribute DOCUMENT
    //   if document is already in list, add image
    //   if image is already in list, do nothing
    //   store or replace obj name for every addObject to be able
    //   to delete a document in deleteobject
    bool add(const char *documentName, const char *imageName);
    void remove(const char *objName);
    void setVisible(const char *documentName, bool visible);
    void setPageNo(const char *documentName, int pageNo);
    void setPosition(const char *documentName, float x, float y, float z);
    void setSize(const char *documentName, int pageNo, float hsize, float vsize);
    void setScale(const char *documentName, float s);
    int findMinPage(const char *documentName);

protected:
    void toggleDocuments(bool visible);
    float vsize_;
    float aspect_ratio_;

    std::map<std::string, coImageViewer *> findCobj_;
    std::map<std::string, coImageViewer *> findDocument_;
    std::string initialObjectName_;

private:
    void menuEvent(coMenuItem *);
};
#endif
