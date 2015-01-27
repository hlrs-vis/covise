/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Tarmac_h
#define Tarmac_h

#include <iostream>
#include <string>

#include "Element.h"
#include "RoadSystemVisitor.h"

class TarmacConnection;

class VEHICLEUTILEXPORT Tarmac : public Element
{
public:
    Tarmac(std::string, std::string);

    void setName(std::string);
    void setId(std::string);

    std::string getName();

    virtual std::string getTypeSpecifier();

    virtual void accept(RoadSystemVisitor *);

protected:
    std::string name;
};

class VEHICLEUTILEXPORT TarmacConnection
{
public:
    TarmacConnection(Tarmac *, int);

    Tarmac *getConnectingTarmac();
    int getConnectingTarmacDirection();

private:
    Tarmac *connectingTarmac;
    int connectingTarmacDirection;
};

#endif
