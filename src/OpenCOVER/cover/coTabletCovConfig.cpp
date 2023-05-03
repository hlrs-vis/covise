
#include "coTabletCovConfig.h"


namespace opencover{

template<typename Ui>
auto getValueFunc(const Ui &ui)
{
    return ui.getValue();
}

template<>
auto getValueFunc(const coTUIToggleButton &ui)
{
    return ui.getState();
}

template<>
auto getValueFunc(const coTUIEditField &ui)
{
    return ui.getText();
}

template<typename Ui, typename ValueType>
void setValueFunc(Ui &ui, const ValueType &val)
{
    ui.setValue(val);
}

template<>
void setValueFunc(coTUIToggleButton &ui, const bool &b)
{
    ui.setState(b);
}

template<>
void setValueFunc(coTUIEditField &ui, const std::string &b)
{
    ui.setText(b);
}

template<typename Ui, typename ValueType>
UiConfigValue<Ui, ValueType>::UiConfigValue(config::File &file, const std::string &section, const std::string &name, coTUIElement *parent, const ValueType& defaultValue, config::Flag flag)
: m_ui(new Ui(name, parent->getID(), defaultValue))
, m_config(file.value(section, name, defaultValue, flag))
{
    setValueFunc(*m_ui, m_config->value());
    m_ui->setEventListener(this);
    m_config->setUpdater([this](const ValueType& val){
        setValueFunc(*m_ui, val);
        if(m_updater)
            m_updater();
    });
}

template<typename Ui, typename ValueType>
ValueType UiConfigValue<Ui, ValueType>::getValue() const
{
    return getValueFunc(*m_ui);
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
void UiConfigValue<Ui, ValueType>::tabletEvent(coTUIElement *tUIItem)
{
    *m_config = getValue();
    if(m_updater)
        m_updater();
}

template class COVEREXPORT UiConfigValue<coTUIToggleButton, bool>;
template class COVEREXPORT UiConfigValue<coTUIEditFloatField, double>;
template class COVEREXPORT UiConfigValue<coTUIEditIntField, int64_t>;
template class COVEREXPORT UiConfigValue<coTUIEditField, std::string>;

template<typename Ui, typename ValueType, size_t Size>
UiConfigValueArray<Ui, ValueType, Size>::UiConfigValueArray(config::File &file, const std::string &section, const std::string &name, coTUIElement *parent, const std::array<ValueType, Size>& defaultValues, config::Flag flag)
: m_config(file.array<ValueType>(section, name, std::vector<ValueType>(defaultValues.begin(), defaultValues.end()), flag))
{
    m_box = new coTUIGroupBox(name, parent->getID());
    auto values = m_config->value();
    for (size_t i = 0; i < Size; i++)
    {
        m_uis[i] = new Ui(name + std::to_string(i), m_box->getID(), values[i]);
        m_uis[i]->setPos((int)i, 0);
        m_uis[i]->setEventListener(this);
    }
    
    m_config->setUpdater([this](){
        auto values = m_config->value();
        for (size_t i = 0; i < Size; i++)
        {
            setValueFunc(*m_uis[i], values[i]);
        }
        if(m_updater)
            m_updater();
    });
}

template<typename Ui, typename ValueType, size_t Size>
    std::array<ValueType, Size> UiConfigValueArray<Ui, ValueType, Size>::getValue() const
{
    auto v = m_config->value();
    std::array<ValueType, Size> a;
    for (size_t i = 0; i < Size; i++)
    {
        a[i] = v[i];
    }
    return a;
}

template<typename Ui, typename ValueType, size_t Size>
void UiConfigValueArray<Ui, ValueType, Size>::setValue(const std::array<ValueType, Size>& val)
{
    for (size_t i = 0; i < Size; i++)
    {
        setValueFunc(*m_uis[i], val[i]);
        *m_config = std::vector<ValueType>(val.begin(), val.end());
    }
}

template<typename Ui, typename ValueType, size_t Size>
void UiConfigValueArray<Ui, ValueType, Size>::setUpdater(std::function<void()> func)
{
    m_updater = func;
}

template<typename Ui, typename ValueType, size_t Size>
std::array<Ui*, Size> &UiConfigValueArray<Ui, ValueType, Size>::uis()
{
    return m_uis;
}

template<typename Ui, typename ValueType, size_t Size>
coTUIGroupBox *UiConfigValueArray<Ui, ValueType, Size>::box()
{
    return m_box;
}

template<typename Ui, typename ValueType, size_t Size>
void UiConfigValueArray<Ui, ValueType, Size>::tabletEvent(coTUIElement *tUIItem)
{
    std::vector<ValueType> v(Size);
    for (size_t i = 0; i < Size; i++)
    {
        v[i] = getValueFunc(*m_uis[i]);
    }
    *m_config = v;
    if(m_updater)
        m_updater();
}

template class COVEREXPORT UiConfigValueArray<coTUIEditFloatField, double, 3>;

}