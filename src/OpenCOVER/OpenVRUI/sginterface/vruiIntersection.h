/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_INTERSECTION_H
#define VRUI_INTERSECTION_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coUpdateManager.h>
#include <list>
#include <string>
#include <vector>

namespace vrui
{

class vruiNode;
class vruiHit;

class OPENVRUIEXPORT vruiIntersection : public coUpdateable
{
public:
    // add this node to the list of intersected nodes
    // call action if node is intersected
    void add(vruiNode *node, coAction *action);

    // keep this node from being intersected
    void remove(vruiNode *node);

    // internal methods (do not call)
    vruiIntersection();
    virtual ~vruiIntersection();

    bool update();

    void remove(coAction *action); // remove all references to this action
    // called by ~coAction

    static vruiIntersection *getIntersectorForAction(const std::string &actionName);
    static vruiIntersection *getIntersector(const std::string &name);

    /// get the Element's classname
    virtual const char *getClassName() const = 0;
    /// get the associated action name
    virtual const char *getActionName() const = 0;

protected:
    virtual void intersect() = 0; // do the intersection

    void callMisses(); // call miss method of all Actions in actionList
    // if node was not intersected in this frame and remove it from the list
    void callActions(vruiNode *node, vruiHit *hit);
    // call all Actions that are attached
    // to that part of the Scenegraph
    //  static unsigned int thisFrame;               // framecounter (used to produce the miss call)

    static std::vector<int *> &frames(); // frame counters for subclasses, subclasses have to push_back a &int to this
    // subclasses have to push_back 'this';
    static std::vector<vruiIntersection *> &intersectors();
    int frameIndex; // subclasses should set this to point to their index in frames

private:
    std::list<coAction *> actionList; // list of intersected Actions
};
}
#endif
