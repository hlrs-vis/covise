#pragma once
#include "vvTabletUI.h"

#include <array>
#include <memory>
#include <functional>
#include <util/coExport.h>

#include "../OpenConfig/array.h"
#include "../OpenConfig/value.h"
#include "../OpenConfig/file.h"

namespace vive
{
template<typename Ui, typename ValueType>
class VVCORE_EXPORT UiConfigValue : private vvTUIListener
{
public:
    UiConfigValue(config::File &file, const std::string &section, const std::string &name, vvTUIElement *parent, const ValueType& defaultValue, config::Flag flag = config::Flag::Default);
    
    ValueType getValue() const;
    void setValue(const ValueType& val);
    void setUpdater(std::function<void()> func);
    Ui *ui();
private:
    void tabletEvent(vvTUIElement *tUIItem) override;

    std::function<void()> m_updater;
    Ui *m_ui;
    std::unique_ptr<ConfigValue<ValueType>> m_config;
};

extern template class VVCORE_EXPORT UiConfigValue<vvTUIToggleButton, bool>;
typedef UiConfigValue<vvTUIToggleButton, bool> covTUIToggleButton;

extern template class VVCORE_EXPORT UiConfigValue<vvTUIEditFloatField, double>;
typedef UiConfigValue<vvTUIEditFloatField, double> covTUIEditFloatField;

extern template class VVCORE_EXPORT UiConfigValue<vvTUIEditIntField, int64_t>;
typedef UiConfigValue<vvTUIEditIntField, int64_t> covTUIEditIntField;

extern template class VVCORE_EXPORT UiConfigValue<vvTUIEditField, std::string>;
typedef UiConfigValue<vvTUIEditField, std::string> covTUIEditField;

template<typename Ui, typename ValueType, size_t Size>
class VVCORE_EXPORT UiConfigValueArray : private vvTUIListener
{
public:
    UiConfigValueArray(config::File &file, const std::string &section, const std::string &name, vvTUIElement *parent, const std::array<ValueType, Size>& defaultValues, config::Flag flag = config::Flag::Default);
    
    std::array<ValueType, Size> getValue() const;
    void setValue(const std::array<ValueType, Size>& val);
    void setUpdater(std::function<void()> func);
    std::array<Ui*, Size> &uis();
    vvTUIGroupBox *box();
private:
    void tabletEvent(vvTUIElement *tUIItem) override;
    vvTUIGroupBox *m_box;
    std::function<void()> m_updater;
    std::array<Ui*, Size> m_uis;
    std::unique_ptr<config::Array<ValueType>> m_config;
};

extern template class VVCORE_EXPORT UiConfigValueArray<vvTUIEditFloatField, double, 3>;
typedef UiConfigValueArray<vvTUIEditFloatField, double, 3> covTUIEditFloatFieldVec3;

}
