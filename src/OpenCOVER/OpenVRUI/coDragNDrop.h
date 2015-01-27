/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DRAGNDROP_H
#define CO_DRAGNDROP_H

#include <list>
#include <OpenVRUI/coUpdateManager.h>

namespace vrui
{

class vruiTransformNode;

// ------------------------------------------------------------------
/// base class for drag'n'drop classes
/// They should be derived from this class and hand over
/// their 'hit's to it.
class OPENVRUIEXPORT coDragNDrop
{
public:
    /// the known media types
    enum
    {
        MEDIA_UNKNOWN = 0,
        MEDIA_MENUITEM
    };

    /// simple constructor
    coDragNDrop();
    /// constructor with specific media type
    coDragNDrop(int);

    // destructor
    virtual ~coDragNDrop();

    int getMediaType()
    {
        return mediaType;
    }

protected:
    /// this objects media type
    int mediaType;

    /// this is the pfDCS that should be used for drag visualisation
    vruiTransformNode *dragNode;

    /// this one checks if drag'n'drop is operated
    /// (like a 'hit()' operation)
    bool processHit();

    /// those two do the work and return if it worked :)
    void dragOperation();

    // this one is supposed to be overloaded and do what
    // the target wants to do!
    // it should return if the operation was sucessful,
    // so the item can be removed or not.
    virtual bool dropOperation(coDragNDrop *) = 0;
};

// ------------------------------------------------------------------
/// This class declares the basic drag'n'drop
/// operations and offers the 'clipboard' to store
/// drag'n'drop data.
class coDragNDropManager : public coUpdateable
{
public:
    coDragNDropManager();
    ~coDragNDropManager();

    /// register this item at the manager. It is updated then.
    void signOn(coDragNDrop *);

    /// sign off from manager
    void signOff(coDragNDrop *);

    /// drag this item into the selection list
    void drag(coDragNDrop *);

    /// drop the item out of the list
    void drop(coDragNDrop *);

    /// deliver the first item which fits on the media type
    coDragNDrop *first(int);

    // pointer on myself as hook
    static coDragNDropManager ddManager;

private:
    // static global storage of grabbed elements
    std::list<coDragNDrop *> selection;

    // static global drag'n'drop classes, for update
    std::list<coDragNDrop *> updateClasses;

    bool update();
};
}
#endif
