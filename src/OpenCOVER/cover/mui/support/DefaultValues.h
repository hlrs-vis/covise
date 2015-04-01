#ifndef MUIDEFAULTVALUES_H
#define MUIDEFAULTVALUES_H

#include <iostream>


namespace mui
{
// Class DefaultValues handles the default-values and returns them to the other functions
// Default-values can easily be changed here
enum AttributesEnum
{
    VisibleEnum = 1,
    ParentEnum = 2,
    LabelEnum = 3,
    DeviceEnum = 4,
    PosXEnum = 5,
    PosYEnum = 6,
    MainWindowEnum =7
};
enum DeviceTypesEnum
{
    TabletEnum = 1,
    CAVEEnum = 2,
    PhoneEnum = 3,
    PowerwallEnum = 4
};
enum UITypeEnum
{
    TUIEnum = 1,
    VRUIEnum = 2
};

std::string getKeywordUI(mui::UITypeEnum UI);

std::string getKeywordDevice(mui::DeviceTypesEnum Device);

std::string getKeywordAttribute(mui::AttributesEnum Attribute);
} // end namespace
#endif
