#include "CityGMLUI.h"
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>

using namespace opencover;

namespace
{

void setButtonStates(const ButtonVec &btns, bool state)
{
    for (auto btn : btns)
        btn->setState(state);
}

BtnCallback makeExclusiveCallback(ButtonVec excludeThese, BtnCallback origCallback)
{
    return [excludeThese = std::move(excludeThese), origCallback](bool on)
    {
        if (on)
            setButtonStates(excludeThese, false);
        origCallback(on);
    };
}
}

CityGMLUI::CityGMLUI(const std::string &name,
    opencover::ui::Menu *parent,
    const Pos &origin)
    : BaseUI(name, parent)
    , m_tab(nullptr)
    , m_pv(nullptr)
{
    assert(parent && "CityGMLUI: Parent is not ready.");
    initUI(name, parent, origin);
    initColorBar();
}


void CityGMLUI::setPVBtnCallback(BtnCallback func)
{
    setBtnCallback(m_pv, func);
}

void CityGMLUI::setColorMapCallback(ColorMapCallback cmc)
{
    m_colorBar->setCallback(cmc);
}

void CityGMLUI::initUI(const std::string &name, opencover::ui::Menu *parent, const Pos &origin)
{
    m_tab = new ui::Menu(parent, "CityGML");

    m_buttons = new opencover::ui::ButtonGroup("GMLSwitch", m_tab);
    for (auto &[name, e] : {
             std::pair { "InfluxCSV", Button::InfluxCSV },
             std::pair { "InfluxArrow", Button::InfluxArrow },
             std::pair { "Static", Button::StaticPower },
             std::pair { "StaticCampus", Button::StaticCampusPower } })
    {
        auto id = static_cast<int>(e);
        m_buttons->add(new ui::Button(m_tab, name, m_buttons, id), id);
    }
    m_pv = new ui::Button(m_tab, "PV");
    m_pv->setText("PV");
    m_pv->setState(true);

    for (auto [type, name, val] : std::array {
             std::tuple { Field::X, "X", origin.x },
             std::tuple { Field::Y, "Y", origin.y },
             std::tuple { Field::Z, "Z", origin.z } })
    {
        m_fields[type] = new ui::EditField(m_tab, name);
        m_fields[type]->setValue(val);
    }
}

void CityGMLUI::initColorBar()
{
    auto menu = new ui::Menu(m_tab, "CityGml_grid");

    m_colorBar = std::make_unique<opencover::CoverColorBar>(menu);
    m_colorBar->setSpecies("Leistung");
    m_colorBar->setUnit("kWh");
    m_colorBar->setName("CityGML");
}
