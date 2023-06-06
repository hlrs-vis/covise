
#include "CovconfigLink.h"

#include "Button.h"
#include "EditField.h"
#include "Element.h"
#include "FileBrowser.h"
#include "Group.h"
#include "Owner.h"
#include "SelectionList.h"
#include "Slider.h"
#include "TextField.h"
namespace opencover{
namespace ui{

template<typename Ui, typename ValueType>
void setValueFunc(Ui &ui, const ValueType &val)
{
    ui.setValue(val);
}

template<>
void setValueFunc(Button &ui, const bool &b)
{
    ui.setState(b);
}

template<>
void setValueFunc(SelectionList &ui, const int64_t &s)
{
   ui.select((int)s);
}

//callback wrappers

template<typename Ui, typename ValueType>
void setCallbackFunc(Ui *ui, std::function<void(ValueType)> cb)
{
    ui->setCallback(cb);
}

template<>
void setCallbackFunc(Slider *ui, std::function<void(double)> cb)
{
    ui->setCallback([cb](double v, bool b){
            cb(v);
    });
}

template<typename Ui, typename ValueType>
UiConfigValue<Ui, ValueType>::UiConfigValue(const std::string &name, Owner *owner, const ValueType& defaultValue, config::File &file, const std::string &section, config::Flag flag)
: m_ui(new Ui(name, owner))
, m_config(file.value(section, name, defaultValue, flag))
{
    init();
}

template<typename Ui, typename ValueType>
UiConfigValue<Ui, ValueType>::UiConfigValue(Group *group, const std::string &name, const ValueType& defaultValue, config::File &file, const std::string &section, config::Flag flag)
: m_ui(new Ui(group, name))
, m_config(file.value(section, name, defaultValue, flag))
{
    init();
}

template<typename Ui, typename ValueType>
void UiConfigValue<Ui, ValueType>::init()
{
    m_initValue = m_config->value();
    setValueFunc(*m_ui, m_config->value());
    std::function<void(ValueType)>f =  [this](const ValueType &val)
                                        {
                                            *m_config = val;
                                            if(m_updater)
                                                m_updater();
                                        };
    setCallbackFunc<Ui, ValueType>(m_ui, f);
    m_config->setUpdater([this](const ValueType& val){
        setValueFunc(*m_ui, val);
        if(m_updater)
            m_updater();
    });
}

template<typename Ui, typename ValueType>
ValueType UiConfigValue<Ui, ValueType>::getValue() const
{
    return m_config->value();
}

template<typename Ui, typename ValueType>
void UiConfigValue<Ui, ValueType>::setValue(const ValueType& val)
{
    setValueFunc(*m_ui, val);
    *m_config =val;
}

template<typename Ui, typename ValueType>
void UiConfigValue<Ui, ValueType>::setUpdater(std::function<void()> func)
{
    m_updater = func;
}

template<typename Ui, typename ValueType>
Ui *UiConfigValue<Ui, ValueType>::ui()
{
    return m_ui;
}

template<typename Ui, typename ValueType>
void UiConfigValue<Ui, ValueType>::restore()
{
    setValue(m_initValue);
}

template class COVEREXPORT UiConfigValue<Button, bool>;
template class COVEREXPORT UiConfigValue<EditField, std::string>;
template class COVEREXPORT UiConfigValue<FileBrowser, std::string>;
template class COVEREXPORT UiConfigValue<SelectionList, int64_t>;
template class COVEREXPORT UiConfigValue<Slider, double>;
template class COVEREXPORT UiConfigValue<TextField, std::string>;

}
}
