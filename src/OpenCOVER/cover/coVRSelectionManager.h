/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_SELECTION_LIST_H
#define COVR_SELECTION_LIST_H
#include <util/common.h>
#include <osg/Node>
#include <OpenVRUI/coUpdateManager.h>

namespace vrui
{
class coNavInteraction;
}
namespace opencover
{
class buttonSpecCell;

class COVEREXPORT coSelectionListener
{
public:
    virtual ~coSelectionListener()
    {
    }
    virtual bool selectionChanged() = 0;
    virtual bool pickedObjChanged() = 0;
};

class COVEREXPORT coVRSelectionManager : public vrui::coUpdateable
{

public:
    coVRSelectionManager();
    ~coVRSelectionManager();
    static coVRSelectionManager *instance();
    void addListener(coSelectionListener *);
    void removeListener(coSelectionListener *);
    void selectionChanged();
    void pickedObjChanged();
    void removeNode(osg::Node *);
    osg::BoundingSphere getBoundingSphere(osg::Node *);

    virtual bool update();

    static void selectionCallback(void *, buttonSpecCell *spec);

    enum HelperNodeType
    {
        MOVE,
        SHOWHIDE,
        SELECTION,
        ANNOTATION
    };

    vrui::coNavInteraction *selectionInteractionA;

    std::list<osg::ref_ptr<osg::Node> > getSelectionList()
    {
        return selectedNodeList;
    };
    std::list<osg::ref_ptr<osg::Group> > getSelectedParentList()
    {
        return selectedParentList;
    };

    void addSelection(osg::Group *parent, osg::Node *selectedNode, bool send = true);
    void receiveAdd(const char *messageData);
    void receiveClear();
    void clearSelection(bool send = true);

    static void insertHelperNode(osg::Group *parent, osg::Node *child, osg::Group *insertNode, HelperNodeType type, bool show = true);
    static osg::Group *getHelperNode(osg::Group *parent, osg::Node *child, HelperNodeType type);
    static bool isHelperNode(osg::Node *);

    static osg::Node *validPath(std::string);
    static std::string generatePath(osg::Node *);
    static std::string generateNames(osg::Node *);

    void setSelectionColor(float R, float G, float B);
    void setSelectionWire(int);
    void setSelectionOnOff(int);
    void showhideSelection(int);

    static void markAsHelperNode(osg::Node *);

private:
    vrui::coUpdateManager *updateManager;

    list<coSelectionListener *> listenerList;

    float SelRed, SelGreen, SelBlue;
    int SelWire;
    int SelOnOff;

    std::list<osg::ref_ptr<osg::Node> > selectedNodeList;
    std::list<osg::ref_ptr<osg::Group> > selectedParentList;
    std::list<osg::ref_ptr<osg::Group> > selectionNodeList;

    static bool hasType(osg::Node *);

    static bool haveToDelete(osg::Node *, osg::Node *);

    static int getHelperType(osg::Node *);
};
}
#endif
