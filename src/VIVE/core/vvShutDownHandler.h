/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __vvShutDownHandler_h
#define __vvShutDownHandler_h

//--------------------------------------------------------------------
// PROJECT        vvShutDownHandler                        © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    BLA BLA BLA BLA BLA BLA
//
// CREATED        20-July-09, S. Franz, U. Woessner
// MODIFIED       21-July-09, S. Franz
//                Application of HLRS style guide
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include <list>
#include <iostream>
namespace vive
{
//--------------------------------------------------------------------
// This class serves as an IFace for all HMIDevice classes for easy and
// controllable shut downs / destruction of singleton classes like
// KLSM, KI, Seat, ...
class vvShutDownHandler
{
public:
    vvShutDownHandler(){};
    virtual ~vvShutDownHandler()
    {
        //std::cerr << "vvShutDownHandlerList::vvShutDownHandlerList" << std::endl;
    }

    virtual void shutDown() = 0;
};
//--------------------------------------------------------------------

//--------------------------------------------------------------------
// This class has at its disposal a list of all classes implementing
// vvShutDownHandler and having theirself added to the list. OPENCover
// calls shutAllDown() in its destructor to destroy all vvShutDownHandler
// singletons and deletes the instance of vvShutDownHandlerList
// (singleton as well) subsequently.
class vvShutDownHandlerList
{
public:
    virtual ~vvShutDownHandlerList();

    void addHandler(vvShutDownHandler *handler);
    void shutAllDown();

    static vvShutDownHandlerList *instance(); // singleton

protected:
    vvShutDownHandlerList();

    static vvShutDownHandlerList *p_sdhl;

private:
    std::list<vvShutDownHandler *> *p_handlerList; // contains NO copies!
};
//--------------------------------------------------------------------
}
#endif
