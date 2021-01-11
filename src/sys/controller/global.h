
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_GLOBAL_H_
#define CTRL_GLOBAL_H_

#include <memory>

#include "controlProcess.h"
#include "object.h"
#include "modui.h"

namespace covise
{
namespace controller
{
    
/**
 * CTRLGlobal singleton class:
 * Stores references to main data objects used by controller
 */
class CTRLGlobal
{
public:
    // reference to the controller object
    std::unique_ptr<Controller> controller;
    // reference to the list of data objects
    std::unique_ptr<object_list> objectList;

    // reference to the list of ui objects
    std::unique_ptr<modui_list> modUIList;
    /** Get the object reference for
       *  accessing the internal data
       *
       *  @return   object reference
       */
    static CTRLGlobal *getInstance();
    int s_nodeID;


    CTRLGlobal();


protected:
private:
    // singleton object used to access
    // references to main controller objects
    static CTRLGlobal *m_global;

    /// copy constructor : assert(0)
    CTRLGlobal(const CTRLGlobal &);
};
} // namespace controller
} // namespace covise
#endif // _CTRL_GLOBAL_H_
