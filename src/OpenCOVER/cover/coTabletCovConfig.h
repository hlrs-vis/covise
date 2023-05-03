#ifndef COTABLETCOVCONFIG_H
#define COTABLETCOVCONFIG_H
#include "coTabletUI.h"

#include <array>
#include <memory>
#include <functional>
#include <util/coExport.h>

#include <OpenConfig/array.h>
#include <OpenConfig/value.h>
#include <OpenConfig/file.h>

namespace opencover
{
template<typename Ui, typename ValueType>
class COVEREXPORT UiConfigValue : private coTUIListener
{
public:
    UiConfigValue(config::File &file, const std::string &section, const std::string &name, coTUIElement *parent, const ValueType& defaultValue, config::Flag flag = config::Flag::Default);
    
    ValueType getValue() const;
    void setValue(const ValueType& val);
    void setUpdater(std::function<void()> func);
    Ui *ui();
private:
    void tabletEvent(coTUIElement *tUIItem) override;

    std::function<void()> m_updater;
    Ui *m_ui;
    std::unique_ptr<ConfigValue<ValueType>> m_config;
};

extern template class COVEREXPORT UiConfigValue<coTUIToggleButton, bool>;
typedef UiConfigValue<coTUIToggleButton, bool> covTUIToggleButton;

extern template class COVEREXPORT UiConfigValue<coTUIEditFloatField, double>;
typedef UiConfigValue<coTUIEditFloatField, double> covTUIEditFloatField;

extern template class COVEREXPORT UiConfigValue<coTUIEditIntField, int64_t>;
typedef UiConfigValue<coTUIEditIntField, int64_t> covTUIEditIntField;

extern template class COVEREXPORT UiConfigValue<coTUIEditField, std::string>;
typedef UiConfigValue<coTUIEditField, std::string> covTUIEditField;

template<typename Ui, typename ValueType, size_t Size>
class COVEREXPORT UiConfigValueArray : private coTUIListener
{
public:
    UiConfigValueArray(config::File &file, const std::string &section, const std::string &name, coTUIElement *parent, const std::array<ValueType, Size>& defaultValues, config::Flag flag = config::Flag::Default);
    
    std::array<ValueType, Size> getValue() const;
    void setValue(const std::array<ValueType, Size>& val);
    void setUpdater(std::function<void()> func);
    std::array<Ui*, Size> &uis();
    coTUIGroupBox *box();
private:
    void tabletEvent(coTUIElement *tUIItem) override;
    coTUIGroupBox *m_box;
    std::function<void()> m_updater;
    std::array<Ui*, Size> m_uis;
    std::unique_ptr<config::Array<ValueType>> m_config;
};

extern template class COVEREXPORT UiConfigValueArray<coTUIEditFloatField, double, 3>;
typedef UiConfigValueArray<coTUIEditFloatField, double, 3> covTUIEditFloatFieldVec3;

}

#endif //COTABLETCOVCONFIG_H