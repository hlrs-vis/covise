//-*-C++-*-
#ifndef COCONFIGVALUEINL
#define COCONFIGVALUEINL

//#include "coConfig.h"


#include "coConfigGroup.h"
#include "coConfigLog.h"

namespace covise
{
class coConfig;

template <class T>
coConfigValue<T>::coConfigValue(const std::string &configGroupName, const std::string &variable, const std::string &section)
    : coConfigValue(variable, section)
{
   this->configGroupName = configGroupName;
}

template <class T>
coConfigValue<T>::coConfigValue(const std::string &variable, const std::string &section)
    : variable(variable), section(section), autoUpdate(false), modified(false), saveToGroup(0), group(0)
{
}

template <class T>
coConfigValue<T>::coConfigValue(const std::string &simpleVariable)
    : coConfigValue("value", simpleVariable)
{
}

template <class T>
coConfigValue<T>::coConfigValue(coConfigGroup *group, const std::string &simpleVariable)
    : coConfigValue(group, "value", simpleVariable)
{
}

template <class T>
coConfigValue<T>::coConfigValue(coConfigGroup *group,
                                const std::string &variable, const std::string &section)
    : coConfigValue(variable, section)
{
   this->saveToGroup = group;
   this->group = group;
}

template <class T>
coConfigValue<T>::coConfigValue(const coConfigValue<T> &value)
    : value(value.value), variable(value.variable), section(value.section), configGroupName(value.configGroupName), autoUpdate(value.autoUpdate), modified(value.modified), unmodifiedValue(value.unmodifiedValue), saveToGroup(value.saveToGroup), group(value.group)
#ifdef COCONFIGVALUE_USE_CACHE
      ,
      cache(value.cache)
#endif
{
   // COCONFIGLOG("coConfigValue<T>::<init> info: copy");
}

template <class T>
coConfigValue<T> & coConfigValue<T>::operator=(const T & rhs)
{

   COCONFIGDBG("coConfigValue<T>::operator= info: " << section << "." << variable << " = " << rhs);

   if (isAutoUpdate() && !modified)
   {
      update();
   }

   modified = true;

   value = toString(rhs);

   //std::cerr << "coConfigValue<T>::operator=T info: " << variable << " in " << section << " = " << value << std::endl;
   if (saveToGroup)
      saveToGroup->setValue(variable, value, section);
   else if (!configGroupName.empty())
   {
      //std::cerr << "coConfigValue<T>::operator=T info: setting in group " << configGroupName << std::endl;
      coConfig::getInstance()->setValueInConfig(variable, value, section, configGroupName);
   }
   else
   {
      coConfig::getInstance()->setValue(variable, value, section);
   }

#ifdef COCONFIGVALUE_USE_CACHE
   cache = rhs;
#endif

   return *this;

}


template <class T>
coConfigValue<T>::operator T()
{
   //COCONFIGLOG("coConfigValue<T>::operator T info: called");
   if (isAutoUpdate()) update();
#ifdef COCONFIGVALUE_USE_CACHE
   return cache;
#else
   return fromString(value);
#endif
}


template <class T>
void coConfigValue<T>::update()
{

   if (group)
   {
      value = group->getValue(variable, section).entry;
      //std::cerr << "coConfigValue<T>::update info: group value " << section << "." << variable << " = " << (value.isNull() ? "*NULL*" : value) << std::endl;
   }
   else
   {
      value = coConfig::getInstance()->getValue(variable, section).entry;
      //COCONFIGLOG("coConfigValue<T>::update info: value " << section << "." << variable << " = " << (value.isNull() ? "*NULL*" : value));
   }

   if (!modified) unmodifiedValue = value;

#ifdef COCONFIGVALUE_USE_CACHE
   cache = fromString(value);
#endif

}


template <class T>
void coConfigValue<T>::setAutoUpdate(bool update)
{
   autoUpdate = update;
}


template <class T>
bool coConfigValue<T>::isAutoUpdate() const
{
   //std::cerr << "coConfigValue<T>::isAutoUpdate() info: " << (autoUpdate ? "true" : "false") << std::endl;
   return autoUpdate;
}


template <class T>
bool coConfigValue<T>::hasValidValue()
{
   // if (isAutoUpdate()) update();
   // COCONFIGLOG("coConfigValue<T>::hasValidValue info: value " << section << "." << variable << " = " << (value.isNull() ? "*NULL*" : value));
   return !value.empty();
}


template <class T>
bool coConfigValue<T>::hasValidValue() const
{
   //COCONFIGLOG("coConfigValue<T>::hasValidValue info: value " << section << "." << variable << " = " << (value.isNull() ? "*NULL*" : value));
   return !value.empty();
}


template <class T>
void coConfigValue<T>::setSaveToGroup(coConfigGroup * group)
{
   saveToGroup = group;
}


template <class T>
coConfigGroup * coConfigValue<T>::getSaveToGroup() const
{
   return saveToGroup;
}


template <class T>
bool coConfigValue<T>::operator==(const coConfigValue<T> & second)
{
   //std::cerr << "coConfigValue<T>::operator info: " << value << "(" << fromString(value) << ") " << second.value << "(" << second.fromString(second.value) << ")" << std::endl;
   if (isAutoUpdate()) update();
#ifdef COCONFIGVALUE_USE_CACHE
   return cache == second.cache;
#else
   return fromString(value) == second.fromString(second.value);
#endif
}


template <class T>
bool coConfigValue<T>::operator!=(const coConfigValue<T> & second)
{
   return ! (*this == second);
}


template <class T>
bool coConfigValue<T>::isModified() const
{
   return modified;
}
}
#endif
