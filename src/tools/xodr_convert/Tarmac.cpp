/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Tarmac.h"

Tarmac::Tarmac(std::string id, std::string n)
    : Element(id)
    , name(n)
{
}

void Tarmac::setName(std::string n)
{
    name = n;
}
void Tarmac::setId(std::string i)
{
    id = i;
}

std::string Tarmac::getName()
{
    return name;
}

TarmacConnection::TarmacConnection(Tarmac *tarmac, int dir)
{
    connectingTarmac = tarmac;
    connectingTarmacDirection = dir;
}

Tarmac *TarmacConnection::getConnectingTarmac()
{
    return connectingTarmac;
}

int TarmacConnection::getConnectingTarmacDirection()
{
    return connectingTarmacDirection;
}
