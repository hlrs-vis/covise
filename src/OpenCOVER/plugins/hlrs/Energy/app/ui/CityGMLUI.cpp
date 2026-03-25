#include "CityGMLUI.h"
#include "app/cover/ui/CoverMenu.h"
#include "lib/core/interfaces/ui/IButton.h"
#include <cover/ui/Menu.h>
#include <memory>

using namespace core::interface::ui;

CityGMLUI::CityGMLUI(const IGUIFactory &factory, const std::string &name,
    IComponent *parent,
    const Pos &origin)
    : BaseUI(name, parent)
    , m_tab(nullptr)
    , m_pv(nullptr)
{
    assert(parent && "CityGMLUI: Parent is not ready.");
    initUI(factory, name, parent, origin);
    initColorBar();
}


void CityGMLUI::setPVBtnCallback(BtnCallback func)
{
    setBtnCallback(m_pv.get(), func);
}

void CityGMLUI::setColorMapCallback(ColorMapCallback cmc)
{
    m_colorBar->setCallback(cmc);
}

void CityGMLUI::initUI(const IGUIFactory &factory, const std::string &name, IComponent *parent, const Pos &origin)
{
    m_tab = factory.createMenu("CityGML", parent);
    std::vector<std::unique_ptr<IButton>> buttons(4);
    for (auto &[name, e] : {
             std::pair { "InfluxCSV", Button::InfluxCSV },
             std::pair { "InfluxArrow", Button::InfluxArrow },
             std::pair { "Static", Button::StaticPower },
             std::pair { "StaticCampus", Button::StaticCampusPower } })
    {
        auto id = static_cast<int>(e);
        buttons[id] = factory.createButton(m_tab.get(), name);
    }

    m_buttons = factory.createButtonGroup(m_tab.get(), "GMLSwitch", std::move(buttons));

    m_pv = factory.createButton(m_tab.get(), "PV");
    m_pv->setText("PV");
    m_pv->setState(true);

    for (auto [type, name, val] : std::array {
             std::tuple { Field::X, "X", origin.x },
             std::tuple { Field::Y, "Y", origin.y },
             std::tuple { Field::Z, "Z", origin.z } })
    {
        m_fields[type] = factory.createEditField(m_tab.get(), name);
        m_fields[type]->setValue(val);
    }
}

void CityGMLUI::initColorBar()
{
    auto c_menu = dynamic_cast<CoverMenu*>(m_tab.get());
    auto menu = new opencover::ui::Menu(c_menu->getMenu(), "CityGml_grid");

    m_colorBar = std::make_unique<opencover::CoverColorBar>(menu);
    m_colorBar->setSpecies("Leistung");
    m_colorBar->setUnit("kWh");
    m_colorBar->setName("CityGML");
}
