/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SGBROWSER_PLUGIN_H
#define _SGBROWSER_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2007/8 HLRS **
 **                                                                           **
 ** Description: Scenegraph Browser											            **
 **																		                     **
 **                                                                           **
 ** Author: Mario Baalcke	                                                   **
 **                                                                           **
 ** History:  								                                          **
 ** Jun-07   v1	    				       		                                 **
 ** April-08 v2                                                               **
 **                                                                           **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRSelectionManager.h>
#include <cover/coVRShader.h>

#include "vrml97/vrml/VrmlNodeTexture.h"
#include "vrml97/vrml/VrmlMFString.h"
#include "vrml97/vrml/VrmlSFBool.h"
#include "vrml97/vrml/VrmlSFInt.h"

#include "vrml97/vrml/Viewer.h"
#include "cover/coTabletUI.h"

#include <util/coTabletUIMessages.h>
#include <util/coRestraint.h>

struct LItem
{
    osg::Image *image;
    int index;
};

class MyNodeVisitor : public osg::NodeVisitor
{
public:
    MyNodeVisitor(TraversalMode tm, coTUISGBrowserTab *sGBrowserTab);
    void apply(osg::Node &node);

    void updateMyParent(osg::Node &node);
    void updateMyChild(osg::Node *node);
    void myUpdate(osg::Node *node);
    void traverseMyParents(osg::Node *node);
    void traverseFindList();
    void sendMyFindList();

    void addMyNode();
    void myInit();

protected:
    coTUISGBrowserTab *sGBrowserTab;
    coVRSelectionManager *selectionManager;

private:
    osg::Node *ROOT;
    osg::Node *selectNode;
    osg::Node *selectParentNode;
    osg::Group *mySelGroupNode;

    std::list<osg::Node *> selNodeList;
    std::list<osg::Node *> selParentList;
    std::list<osg::Group *> selGroupList;
    std::list<osg::Node *> findNodeList;
    std::list<osg::Node *> sendFindList;
    std::list<osg::Node *> sendCurrentList;
};
class TexVisitor : public osg::NodeVisitor
{
public:
    TexVisitor(TraversalMode tm, coTUISGBrowserTab *TextureTab);
    ~TexVisitor();
    void apply(osg::Node &node);
    void clearImageList()
    {
        imageListe.clear();
    };
    void insertImage(LItem item)
    {
        imageListe.push_back(item);
    };
    int findImage(osg::Image *);
    LItem *getLItem(osg::Image *);
    int getListSize()
    {
        return imageListe.size();
    };
    bool getTexFound()
    {
        return texFound;
    };
    void setTexFound(bool state)
    {
        texFound = state;
    };
    int getMax();
    void processStateSet(osg::StateSet *ss);
    osg::Image *getImage(int index);
    void sendImageList();

protected:
    coTUISGBrowserTab *texTab;
    std::vector<LItem> imageListe;

private:
    bool texFound;
};

class SGBrowser : public coVRPlugin, public coTUIListener, public coSelectionListener
{
public:
    SGBrowser();
    virtual ~SGBrowser();
    bool init();

    virtual bool selectionChanged();
    virtual bool pickedObjChanged();
    //_____________________________this will be called in PreFrame_____________________________
    void preFrame();
    void message(int toWhom, int type, int len, const void *buf);
    void removeNode(osg::Node *node, bool isGroup, osg::Node *realNode);
    void addNode(osg::Node *node, const RenderObject *obj);
    bool processTexture(coTUISGBrowserTab *sGBrowserTab, TexVisitor *texvis, osg::StateSet *ss);

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    virtual void tabletSelectEvent(coTUIElement *tUIItem);
    virtual void tabletChangeModeEvent(coTUIElement *tUIItem);
    virtual void tabletFindEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletCurrentEvent(coTUIElement *tUIItem);
    virtual void tabletDataEvent(coTUIElement *tUIItem, TokenBuffer &tb);
    virtual void tabletLoadFilesEvent(char *nodeName);

    //   virtual void hideNode();
    static SGBrowser *plugin;

private:
    char *idata;
    bool myMes;
    bool reconnect;
    //_____________________________the plugin tab__________________________________________________________

    coVRSelectionManager *selectionManager;
#if 0
    std::vector<coTUISGBrowserTab *> sGBrowserTab;
    std::vector<MyNodeVisitor *> vis;
    std::vector<TexVisitor *> texvis;
#endif

    coRestraint *restraint;
    coVRShaderList *shaderList;

    struct TuiData
    {
        TuiData(coTUISGBrowserTab *tab, MyNodeVisitor *vis, TexVisitor *tex)
            : tab(tab), vis(vis), tex(tex)
        {}

        coTUISGBrowserTab *tab;
        MyNodeVisitor *vis;
        TexVisitor *tex;
    };

    std::vector<TuiData> tuis;

    bool linked;

    osg::ref_ptr<osg::Node> pickedObject;
};
#endif
