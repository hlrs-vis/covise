#include "DefaultValues.h"

using namespace mui;


std::string mui::getKeywordUI(UITypeEnum UI)
{
    switch (UI)
    {
    case mui::TUIEnum:
        return "TUI";
    case mui::VRUIEnum:
        return "VRUI";
    }
    std::cerr << "ConfigParser::getKeywordUI(): UITypeEnum number " << UI << " not known" << std::endl;
    return "";
}

std::string mui::getKeywordDevice(DeviceTypesEnum Device)
{
    switch (Device)
    {
    case mui::TabletEnum:
        return "Tablet";
    case mui::CAVEEnum:
        return "CAVE";
    case mui::PhoneEnum:
        return "Phone";
    case mui::PowerwallEnum:
        return "Powerwall";
    }
    std::cerr << "ConfigParser::getKeywordDevice(): DeviceTypeEnum number " << Device << " not known" << std::endl;
    return "";
}

std::string mui::getKeywordAttribute(AttributesEnum Attribute)
{
    switch (Attribute)
    {
    case mui::VisibleEnum:
        return "visible";
    case mui::ParentEnum:
        return "parent";
    case mui::LabelEnum:
        return "label";
    case mui::DeviceEnum:
        return "device";
    case mui::PosXEnum:
        return "posx";
    case mui::PosYEnum:
        return "posy";
    case mui::MainWindowEnum:
        return "mainwindow";
    }

    std::cerr << "ConfigParser::getKeywordUI(): AttributesEnum number " << Attribute << " not known" << std::endl;
    return "";
}
