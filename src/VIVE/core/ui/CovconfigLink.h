#pragma once

#include <memory>
#include <functional>
#include <util/coExport.h>

#include "../../OpenConfig/array.h"
#include "../../OpenConfig/value.h"
#include "../../OpenConfig/file.h"


namespace vive
{
namespace ui {

class Button;
class EditField;
class Element;
class FileBrowser;
class Group;
class Owner;
class SelectionList;
class Slider;
class TextField;


template<typename Ui, typename ValueType>
class VVCORE_EXPORT UiConfigValue
{
public:
    UiConfigValue(const std::string &name, Owner *owner, const ValueType& defaultValue, config::File &file, const std::string &section, config::Flag flag = config::Flag::Default);
    UiConfigValue(Group *group, const std::string &name, const ValueType& defaultValue, config::File &file, const std::string &section, config::Flag flag = config::Flag::Default);
    
    ValueType getValue() const;
    void setValue(const ValueType& val);
    void setUpdater(std::function<void()> func);
    Ui *ui(); // warning: some ui function can change the value of the ui field without updating the config value
    void restore(); // restore config value
private:
    void init();
    std::function<void()> m_updater;
    Ui *m_ui;
    std::unique_ptr<ConfigValue<ValueType>> m_config;
    ValueType m_initValue;
};

extern template class VVCORE_EXPORT UiConfigValue<Button, bool>;
typedef UiConfigValue<Button, bool> ButtonConfigValue;

extern template class VVCORE_EXPORT UiConfigValue<EditField, std::string>;
typedef UiConfigValue<EditField, std::string> EditFieldConfigValue;

extern template class VVCORE_EXPORT UiConfigValue<FileBrowser, std::string>;
typedef UiConfigValue<FileBrowser, std::string> FileBrowserConfigValue;

extern template class VVCORE_EXPORT UiConfigValue<SelectionList, int64_t>;
typedef UiConfigValue<SelectionList, int64_t> SelectionListConfigValue;

extern template class VVCORE_EXPORT UiConfigValue<Slider, double>;
typedef UiConfigValue<Slider, double> SliderConfigValue;

extern template class VVCORE_EXPORT UiConfigValue<TextField, std::string>;
typedef UiConfigValue<TextField, std::string> TextFieldConfigValue;


}



}
