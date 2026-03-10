#pragma once

#include "BaseUI.h"
#include <PluginUtil/colors/ColorBar.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <memory>
#include <string>
#include <map>
#include "app/typedefs.h"

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
        return dynamic_cast<opencover::ui::Button *>(m_buttons->child(static_cast<int>(button)));
    }

    auto getEditField(Field field)
    {
        return m_fields[field];
    }

public:
    CityGMLUI(const std::string &name, opencover::ui::Menu *parent, const Pos &origin = { 0.0f, 0.0f, 0.0f });
    void setColorMapCallback(ColorMapCallback cm);

    auto InfluxCSV() { return getButton(Button::InfluxCSV); }
    auto InfluxArrow() { return getButton(Button::InfluxArrow); }
    auto PV() { return m_pv; }
    auto StaticCampusPower() { return getButton(Button::StaticCampusPower); }
    auto StaticPower() { return getButton(Button::StaticPower); }
    auto getTranslation() { return Pos { getEditField(Field::X)->number(), getEditField(Field::Y)->number(), getEditField(Field::Z)->number() }; }
    opencover::CoverColorBar *colorBar() { return m_colorBar.get(); }
    void setEditFieldCallback(Field field, const EditCallback &func) { setTxtFieldCallback(getEditField(field), func); }
    auto &getEditFields() { return m_fields; }
    void setPVBtnCallback(BtnCallback func);
    void setButtonGroupCallback(BtnCallback func) { m_buttons->setCallback(func); }

private:
    void initUI(const std::string &name, opencover::ui::Menu *parent, const Pos &origin);
    void initColorBar();
    std::unique_ptr<opencover::CoverColorBar> m_colorBar;

    // deleted by owner => DON'T DELETE IN DESTRUCTOR
    opencover::ui::Menu *m_tab;
    std::map<Field, opencover::ui::EditField *> m_fields;
    opencover::ui::ButtonGroup *m_buttons;
    opencover::ui::Button *m_pv;
};
