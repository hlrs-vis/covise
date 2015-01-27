/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UPDATE_MANAGER_H
#define CO_UPDATE_MANAGER_H

#include <util/coTypes.h>

#include <list>

namespace vrui
{

/// objects that are derived from this class are called once per frame
class OPENVRUIEXPORT coUpdateable
{
public:
    coUpdateable(); ///< Constructor
    /** you should adds this object to the update manager in your constructor
      */
    virtual ~coUpdateable(); ///< Destructor
    /** this method is called once per frame as
          long as it returns true.
          as soon as it returns false, it is removed from the update manager
          and not called again.
          the update method is called prior to preFrame();
      */
    virtual bool update() = 0;
};

/** the UpdateManager keeps a list of all updateable objects and
    calls their update method once per frame.
*/

//template class OPENVRUIEXPORT std::list<coUpdateable *>;

class OPENVRUIEXPORT coUpdateManager
{
protected:
    std::list<coUpdateable *> updateList; ///< list of updateable objects

public:
    coUpdateManager(); ///< Constructor
    ~coUpdateManager(); ///< Desctructor
    void update(); ///< all objectpreFrame
    void add(coUpdateable *, bool first = false); ///< add an updateable object
    void remove(coUpdateable *); ///< remove an updateable object
    void removeAll(); ///< remove all updateable object
};
}
#endif
