#ifndef MUIDEFAULTVALUES_H
#define MUIDEFAULTVALUES_H

#include <iostream>


namespace mui
{
// Class DefaultValues handles the default-values and returns them to the other functions
// Default-values can easily be changed here
enum AttributesEnum
{
    VisibleEnum = 0,
    ParentEnum = 1,
    LabelEnum = 2,
    DeviceEnum = 3,
    PosXEnum = 4,
    PosYEnum = 5
};
enum DeviceTypesEnum
{
    TabletEnum = 0,
    CAVEEnum = 1,
    PhoneEnum = 2,
    PowerwallEnum = 3,

    muiDeviceCounter = 4        // last element of this Enum; needet to loop over all enums
};
enum UITypeEnum
{
    TUIEnum = 0,
    VRUIEnum = 1,

    muiUICounter = 2            // last element of this Enum; needet to loop over all enums
};

std::string getKeywordUI(mui::UITypeEnum UI);

std::string getKeywordDevice(mui::DeviceTypesEnum Device);

std::string getKeywordAttribute(mui::AttributesEnum Attribute);
} // end namespace
#endif
