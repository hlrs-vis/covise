#pragma once

#include "BaseUI.h"
#include "app/typedefs.h"
#include <PluginUtil/ColorBar.h>
#include <memory>
#include <string>
#include <map>
#include <lib/core/interfaces/ui/IGUIFactory.h>
#include <lib/core/interfaces/ui/IComponent.h>
#include <lib/core/interfaces/ui/IButton.h>
#include <lib/core/interfaces/ui/IButtonGroup.h>
#include <lib/core/interfaces/ui/IEditField.h>
#include <lib/core/interfaces/ui/IMenu.h>

typedef std::function<void(const opencover::ColorMap &)> ColorMapCallback;
typedef std::vector<opencover::ui::Button *> ButtonVec;

class CityGMLUI : BaseUI
{
    enum class Button
    {
        InfluxCSV,
        InfluxArrow,
        StaticPower,
        StaticCampusPower
    };

    enum class Field
    {
        X,
        Y,
        Z
    };

    auto getButton(Button button)
    {
        return m_buttons->getChild(static_cast<int>(button));
    }

    auto getEditField(Field field)
    {
        return m_fields[field].get();
    }

public:
    CityGMLUI(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent, const Pos &origin = { 0.0f, 0.0f, 0.0f });
    void setColorMapCallback(ColorMapCallback cm);

    auto InfluxCSV() { return getButton(Button::InfluxCSV); }
    auto InfluxArrow() { return getButton(Button::InfluxArrow); }
    auto PV() { return m_pv.get(); }
    auto StaticCampusPower() { return getButton(Button::StaticCampusPower); }
    auto StaticPower() { return getButton(Button::StaticPower); }
    auto getTranslation() { return Pos { getEditField(Field::X)->getValue(), getEditField(Field::Y)->getValue(), getEditField(Field::Z)->getValue() }; }
    opencover::CoverColorBar *colorBar() { return m_colorBar.get(); }
    void setEditFieldCallback(Field field, const EditCallback &func) { setTxtFieldCallback(getEditField(field), func); }
    auto &getEditFields() { return m_fields; }
    void setPVBtnCallback(BtnCallback func);
    void setButtonGroupCallback(BtnCallback func) { m_buttons->setCallback(func); }

private:
    void initUI(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent, const Pos &origin);
    void initColorBar();
    // TODO: rewrite this later
    std::unique_ptr<opencover::CoverColorBar> m_colorBar;

    std::unique_ptr<core::interface::ui::IMenu> m_tab;
    std::map<Field, std::unique_ptr<core::interface::ui::IEditDoubleField>> m_fields;
    std::unique_ptr<core::interface::ui::IButtonGroup> m_buttons;
    std::unique_ptr<core::interface::ui::IButton> m_pv;
};
