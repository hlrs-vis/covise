/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ACTION_H
#define CO_ACTION_H

#include <util/coTypes.h>
#include <string>

namespace vrui
{

class vruiHit;
class vruiNode;
class vruiActionUserData;

/**
    classes that are derived from coAction can be attached tho the
    scenegraph and will be called when user interacts with the object
*/
class OPENVRUIEXPORT coAction
{
public:
    coAction(); ///< Constructor
    virtual ~coAction(); ///< Destructor

    /** hit is called whenever the node, or any node underneath the node
          with this action is intersected
          return ACTION_CALL_ON_MISS if you want miss to be called
          otherwise return ACTION_DONE
      */
    virtual int hit(vruiHit *hit) = 0;

    /** miss is called once after a hit, if the node is not intersected
          anymore
      */
    virtual void miss();

    enum Result
    {
        ACTION_DONE = 0x00,
        ACTION_CALL_ON_MISS = 0x01,
        ACTION_UNDEF = 0x02
    };

    // internal:
    /// INTERNAL set the framecounter (used to produce the miss call)
    void setFrame(unsigned int frame)
    {
        thisFrame = frame;
    }
    /// INTERNAL get the framecounter (used to produce the miss call)
    unsigned int getFrame() const
    {
        return thisFrame;
    }
    /// INTERNAL set the node this action belongs to
    void setNode(vruiNode *);
    /// INTERNAL add a child action
    void addChild(coAction *);
    /// INTERNAL same as miss/hit, but do it for all children as well
    virtual int hitAll(vruiHit *hit);
    /// INTERNAL same as miss/hit, but do it for all children as well
    virtual void missAll();

private:
    vruiActionUserData *userData; ///< userdata object attached to nodes in the scengraph
    vruiNode *myNode; ///< the node in the scengraph
    coAction *child; ///< children of this action
    coAction *parent; ///< parent action
    unsigned int thisFrame; ///< framecounter (used to produce the miss call)

protected:
    /** name of this action this can be used to distinguish betwenn different types of actions
          like touch and intersection actions
      */
    std::string actionName;
};

}
#endif
