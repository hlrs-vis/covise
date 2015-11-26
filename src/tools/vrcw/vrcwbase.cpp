#include "vrcwbase.h"


/*****
 * constructor - destructor
 *****/

VRCWBase::VRCWBase()
{

}

VRCWBase::~VRCWBase()
{

}


/*****
 * public functions
 *****/
int VRCWBase::processGuiInput(const QList<VRCWBase*>& vrcwList)
{
    if (vrcwList.size() > 0)
    {
        return 999999;
    }
    else
    {
        return 999998;
    }
}

int VRCWBase::processGuiInput(const int& index,
        const QList<VRCWBase*>& vrcwList)
{
    if (vrcwList.size() >= index)
    {
        return 9999999 + index;
    }
    else
    {
        return 9999998;
    }
}
