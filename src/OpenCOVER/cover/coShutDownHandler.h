/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __coShutDownHandler_h
#define __coShutDownHandler_h

//--------------------------------------------------------------------
// PROJECT        coShutDownHandler                        © 2009 HLRS
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
namespace opencover
{
//--------------------------------------------------------------------
// This class serves as an IFace for all HMIDevice classes for easy and
// controllable shut downs / destruction of singleton classes like
// KLSM, KI, Seat, ...
class coShutDownHandler
{
public:
    coShutDownHandler(){};
    virtual ~coShutDownHandler()
    {
        //std::cerr << "coShutDownHandlerList::coShutDownHandlerList" << std::endl;
    }

    virtual void shutDown() = 0;
};
//--------------------------------------------------------------------

//--------------------------------------------------------------------
// This class has at its disposal a list of all classes implementing
// coShutDownHandler and having theirself added to the list. OPENCover
// calls shutAllDown() in its destructor to destroy all coShutDownHandler
// singletons and deletes the instance of coShutDownHandlerList
// (singleton as well) subsequently.
class coShutDownHandlerList
{
public:
    virtual ~coShutDownHandlerList();

    void addHandler(coShutDownHandler *handler);
    void shutAllDown();

    static coShutDownHandlerList *instance(); // singleton

protected:
    coShutDownHandlerList();

    static coShutDownHandlerList *p_sdhl;

private:
    std::list<coShutDownHandler *> *p_handlerList; // contains NO copies!
};
//--------------------------------------------------------------------
}
#endif
