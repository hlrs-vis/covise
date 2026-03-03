#include "CityGMLUI.h"
#include "app/cover/ui/BaseUI.h"
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <vector>

using namespace opencover;

namespace
{

typedef std::vector<opencover::ui::Button *> Buttons;
void setButtonStates(const Buttons &btns, bool state)
{
    std::for_each(btns.begin(), btns.end(), [&](opencover::ui::Button *btn)
        { btn->setState(state); });
}

BtnCallback disableButtonFuncWrapper(Buttons btns, BtnCallback callback)
{
    return [btns = std::move(btns), callback](bool on)
    {
        if (on && !btns.empty())
            setButtonStates(btns, false);
        callback(on);
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
    auto btnWrap = disableButtonFuncWrapper({ 
        m_staticPower, 
        m_staticCampusPower, 
        m_enableInfluxArrow 
    }, 
    func);
    setBtnCallback(m_enableInfluxCSV, btnWrap);
}

void CityGMLUI::setInfluxArrowBtnCallback(BtnCallback func) { 
    auto btnWrap = disableButtonFuncWrapper({ 
        m_staticPower, 
        m_staticCampusPower, 
        m_enableInfluxCSV
    }, 
    func);
    setBtnCallback(m_enableInfluxArrow, btnWrap); 
}

void CityGMLUI::setPVBtnCallback(BtnCallback func) { 
    setBtnCallback(m_PVEnable, func); 
}

void CityGMLUI::setStaticPowerBtnCallback(BtnCallback func) { 
    auto btnWrap = disableButtonFuncWrapper({ 
        m_enableInfluxCSV,
        m_enableInfluxArrow,
        m_staticCampusPower
    }, 
    func);
    setBtnCallback(m_staticPower, btnWrap); 
}

void CityGMLUI::setStaticCampusPowerBtnCallback(BtnCallback func) { 
    auto btnWrap = disableButtonFuncWrapper({ 
        m_enableInfluxCSV,
        m_enableInfluxArrow,
        m_staticPower
    }, 
    func);
    setBtnCallback(m_staticCampusPower, btnWrap); 
}

void CityGMLUI::setColorMapCallback(ColorMapCallback cmc) {
   m_colorBar->setCallback(cmc);
}

void CityGMLUI::initUI(const std::string &name, opencover::ui::Menu *parent, const CityGMLOrigin &origin)
{
    m_tab = new ui::Menu(parent, "CityGML");
    // m_tab = new ui::Menu("CityGML", this);
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
