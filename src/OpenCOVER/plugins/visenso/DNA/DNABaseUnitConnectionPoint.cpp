/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DNABaseUnitConnectionPoint.h"
#include "DNABaseUnit.h"

DNABaseUnitConnectionPoint::DNABaseUnitConnectionPoint(DNABaseUnit *myBase, std::string myName, osg::Vec3 point, osg::Vec3 normal, std::string myConnectionName)
    : myBaseUnitName_(myName)
    , connectableBaseUnitName_(myConnectionName)
    , point_(point)
    , normal_(normal)
    , isConnected_(false)
    , isEnabled_(true)
    , rotation_(false)
    , myBaseUnit_(myBase)
    , connectedPoint_(NULL)
{
}

DNABaseUnitConnectionPoint::~DNABaseUnitConnectionPoint()
{
}

bool DNABaseUnitConnectionPoint::connectTo(DNABaseUnitConnectionPoint *connPoint)
{
    if (isEnabled_)
    {
        isConnected_ = true;
        connectedPoint_ = connPoint;
        return true;
    }
    return false;
}

void DNABaseUnitConnectionPoint::disconnectBase()
{
    isConnected_ = false;
    connectedPoint_ = NULL;
}

void DNABaseUnitConnectionPoint::setEnabled(bool b)
{
    // check if we have to do something
    if (isEnabled_ == b)
        return;

    isEnabled_ = b;
    // disconnect the connection
    if (!isEnabled_ && isConnected_)
        myBaseUnit_->setConnection(myBaseUnitName_, connectableBaseUnitName_, false, false, NULL);
}
