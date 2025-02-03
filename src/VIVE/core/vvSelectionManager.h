/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_SELECTION_LIST_H
#define COVR_SELECTION_LIST_H
#include <util/common.h>
#include <vsg/nodes/Node.h>
#include <OpenVRUI/coUpdateManager.h>

namespace vrui
{
class coNavInteraction;
}
namespace covise {
class TokenBuffer;
}
namespace vive
{

class VVCORE_EXPORT coSelectionListener
{
public:
    virtual ~coSelectionListener()
    {
    }
    virtual bool selectionChanged() = 0;
    virtual bool pickedObjChanged() = 0;
};

class VVCORE_EXPORT vvSelectionManager : public vrui::coUpdateable
{
    static vvSelectionManager *s_instance;
    vvSelectionManager();

public:
    ~vvSelectionManager();
    static vvSelectionManager *instance();
    void addListener(coSelectionListener *);
    void removeListener(coSelectionListener *);
    void selectionChanged();
    void pickedObjChanged();
    void removeNode(vsg::Node *);
    /*osg::BoundingSphere getBoundingSphere(vsg::Node*);*/

    virtual bool update();

    enum HelperNodeType
    {
        MOVE,
        SHOWHIDE,
        SELECTION,
        ANNOTATION
    };

    vrui::coNavInteraction *selectionInteractionA;

    std::list<vsg::ref_ptr<vsg::Node> > getSelectionList()
    {
        return selectedNodeList;
    };
    std::list<vsg::ref_ptr<vsg::Group> > getSelectedParentList()
    {
        return selectedParentList;
    };

    void addSelection(const vsg::Group *parent, const vsg::Node *selectedNode, bool send = true);
    void receiveAdd(covise::TokenBuffer &messageData);
    void receiveClear();
    void clearSelection(bool send = true);

    static void insertHelperNode(vsg::Group *parent, vsg::Node *child, vsg::Group *insertNode, HelperNodeType type, bool show = true);
    static vsg::Group *getHelperNode(vsg::Group *parent, vsg::Node *child, HelperNodeType type);
    static bool isHelperNode(const vsg::Node *);

    static vsg::Node *validPath(std::string);
    static std::string generatePath(const vsg::Node *);
    static std::string generateNames(vsg::Node *);

    void setSelectionColor(float R, float G, float B);
    void setSelectionWire(int);
    void setSelectionOnOff(int);
    void showhideSelection(int);

    static void markAsHelperNode(vsg::Node *);

private:
    vrui::coUpdateManager *updateManager;

    list<coSelectionListener *> listenerList;

    float SelRed, SelGreen, SelBlue;
    int SelWire;
    int SelOnOff;

    std::list<vsg::ref_ptr<vsg::Node> > selectedNodeList;
    std::list<vsg::ref_ptr<vsg::Group> > selectedParentList;
    std::list<vsg::ref_ptr<vsg::Group> > selectionNodeList;

    static bool hasType(vsg::Node *);

    static bool haveToDelete(vsg::Node *, vsg::Node *);

    static int getHelperType(vsg::Node *);
};
}
#endif
