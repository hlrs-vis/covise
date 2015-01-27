/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FASI_UPDATE_MANAGER
#define FASI_UPDATE_MANAGER
#include <list>
#include <util/coExport.h>

/// objects that are derived from this class are called once per frame
class VEHICLEUTILEXPORT fasiUpdateable
{
public:
    fasiUpdateable(); ///< Constructor
    virtual ~fasiUpdateable(); ///< Destructor
    virtual bool update()
    {
        return false;
    };
};

class VEHICLEUTILEXPORT fasiUpdateManager
{
protected:
    std::list<fasiUpdateable *> updateList; ///< list of updateable objects

public:
    fasiUpdateManager(); ///< Constructor
    ~fasiUpdateManager(); ///< Desctructor
    void update(); ///< all objectpreFrame
    void add(fasiUpdateable *, bool first = false); ///< add an updateable object
    void remove(fasiUpdateable *); ///< remove an updateable object
    void removeAll(); ///< remove all updateable object
    static fasiUpdateManager *fum;
    static fasiUpdateManager *instance();
};

#endif
