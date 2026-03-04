#include "CityGMLUI.h"
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <initializer_list>

using namespace opencover;

namespace
{

void setButtonStates(std::initializer_list<ui::Button *> btns, bool state)
{
    for (auto btn : btns)
        btn->setState(state);
}

BtnCallback makeExclusiveCallback(std::initializer_list<ui::Button *> excludeThese, BtnCallback origCallback)
{
    return [excludeThese, origCallback](bool on)
    {
        if (on)
            setButtonStates(excludeThese, false);
        origCallback(on);
    };
}
}

CityGMLUI::CityGMLUI(const std::string &name,
    opencover::ui::Menu *parent,
    const CityGMLOrigin &origin)
    : BaseUI(name, parent)
    , m_tab(nullptr)
    , m_enableInfluxCSV(nullptr)
    , m_PVEnable(nullptr)
    , m_enableInfluxArrow(nullptr)
    , m_staticCampusPower(nullptr)
    , m_staticPower(nullptr)
{
    initUI(name, parent, origin);
    initColorBar();
}

void CityGMLUI::setInfluxCSVBtnCallback(BtnCallback func)
{
    setBtnCallback(m_enableInfluxCSV, makeExclusiveCallback({ m_staticPower, m_staticCampusPower, m_enableInfluxArrow }, func));
}

void CityGMLUI::setInfluxArrowBtnCallback(BtnCallback func)
{
    setBtnCallback(m_enableInfluxArrow,
        makeExclusiveCallback({ m_staticPower,
                                  m_staticCampusPower,
                                  m_enableInfluxCSV },
            func));
}

void CityGMLUI::setPVBtnCallback(BtnCallback func)
{
    setBtnCallback(m_PVEnable, func);
}

void CityGMLUI::setStaticPowerBtnCallback(BtnCallback func)
{
    setBtnCallback(m_staticPower, makeExclusiveCallback({ m_enableInfluxCSV, m_enableInfluxArrow, m_staticCampusPower }, func));
}

void CityGMLUI::setStaticCampusPowerBtnCallback(BtnCallback func)
{

    setBtnCallback(m_staticCampusPower, makeExclusiveCallback({ m_enableInfluxCSV, m_enableInfluxArrow, m_staticPower }, func));
}

void CityGMLUI::setColorMapCallback(ColorMapCallback cmc)
{
    m_colorBar->setCallback(cmc);
}

void CityGMLUI::initUI(const std::string &name, opencover::ui::Menu *parent, const CityGMLOrigin &origin)
{
    m_tab = new ui::Menu(parent, "CityGML");
    m_enableInfluxCSV = new ui::Button(m_tab, "InfluxCSV");

    m_enableInfluxArrow = new ui::Button(m_tab, "InfluxArrow");

    m_PVEnable = new ui::Button(m_tab, "PV");
    m_PVEnable->setText("PV");
    m_PVEnable->setState(true);

    m_staticPower = new ui::Button(m_tab, "Static");
    m_staticPower->setText("StaticPower");
    m_staticPower->setState(false);

    m_staticCampusPower = new ui::Button(m_tab, "StaticCampus");
    m_staticCampusPower->setText("StaticPowerCampus");
    m_staticCampusPower->setState(false);

    m_X = new ui::EditField(m_tab, "X");
    m_Y = new ui::EditField(m_tab, "Y");
    m_Z = new ui::EditField(m_tab, "Z");

    m_X->setValue(origin.x);
    m_Y->setValue(origin.y);
    m_Z->setValue(origin.z);
}

void CityGMLUI::initColorBar()
{
    auto menu = new ui::Menu(m_tab, "CityGml_grid");

    m_colorBar = std::make_unique<opencover::CoverColorBar>(menu);
    m_colorBar->setSpecies("Leistung");
    m_colorBar->setUnit("kWh");
    m_colorBar->setName("CityGML");
}
