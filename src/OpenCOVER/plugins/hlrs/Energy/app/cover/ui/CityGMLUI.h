#pragma once

#include "BaseUI.h"
#include <PluginUtil/colors/ColorBar.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <string>

typedef std::function<void(const opencover::ColorMap&)> ColorMapCallback;

struct CityGMLOrigin
{
    CityGMLOrigin(double x, double y, double z)
        : x(x)
        , y(y)
        , z(z)
    {
    }
    double x, y, z;
};

class CityGMLUI : BaseUI
{
public:
    CityGMLUI(const std::string &name, opencover::ui::Menu *parent, const CityGMLOrigin &origin = { 0.0f, 0.0f, 0.0f });
    void setInfluxCSVBtnCallback(BtnCallback func);
    void setInfluxArrowBtnCallback(BtnCallback func);
    void setPVBtnCallback(BtnCallback func);
    void setStaticPowerBtnCallback(BtnCallback func);
    void setStaticCampusPowerBtnCallback(BtnCallback func);
    void setXCallback(EditCallback func) { setTxtFieldCallback(m_X, func); }
    void setYCallback(EditCallback func) { setTxtFieldCallback(m_Y, func); }
    void setZCallback(EditCallback func) { setTxtFieldCallback(m_Z, func); }
    void setColorMapCallback(ColorMapCallback cm);
    auto getTranslation() const { return CityGMLOrigin{ m_X->number(), m_Y->number(), m_Z->number() }; }
    auto getInfluxArrowBtnState() const { return m_enableInfluxArrow->state(); }
    auto getInfluxCSVBtnState() const { return m_enableInfluxCSV->state(); }
    auto getStaticCampusPowerBtnState() const { return m_staticCampusPower->state(); }
    auto getStaticPowerBtnState() const { return m_staticPower->state(); }
    opencover::CoverColorBar *colorBar() { return  m_colorBar.get(); }

private:
    void initUI(const std::string &name, opencover::ui::Menu *parent, const CityGMLOrigin &origin);
    void initColorBar();
    std::unique_ptr<opencover::CoverColorBar> m_colorBar;

    opencover::ui::Menu *m_tab;
    opencover::ui::EditField *m_X;
    opencover::ui::EditField *m_Y;
    opencover::ui::EditField *m_Z;
    opencover::ui::Button *m_enableInfluxCSV;
    opencover::ui::Button *m_enableInfluxArrow;
    opencover::ui::Button *m_PVEnable;
    opencover::ui::Button *m_staticCampusPower;
    opencover::ui::Button *m_staticPower;
};
