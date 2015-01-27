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
coConfigValue<T>::coConfigValue(const QString & configGroupName, const QString & variable, const QString & section)
{

   this->variable = variable;
   this->section = section;
   this->configGroupName = configGroupName;
   this->autoUpdate = false;

   this->modified = false;
   this->unmodifiedValue = value;

   this->saveToGroup = 0;
   this->group = 0;

}


template <class T>
coConfigValue<T>::coConfigValue(const QString & variable, const QString & section)
{

   this->variable = variable;
   this->section = section;
   this->configGroupName = QString::null;
   this->autoUpdate = false;

   this->modified = false;
   this->unmodifiedValue = value;

   this->saveToGroup = 0;
   this->group = 0;

}


template <class T>
coConfigValue<T>::coConfigValue(const QString & simpleVariable)
{

   this->variable = "value";
   this->section = simpleVariable;
   this->configGroupName = QString::null;
   this->autoUpdate = false;

   this->modified = false;
   this->unmodifiedValue = value;

   this->saveToGroup = 0;
   this->group = 0;

}


template <class T>
coConfigValue<T>::coConfigValue(coConfigGroup * group, const QString & simpleVariable)
{

   this->variable = "value";
   this->section = simpleVariable;
   this->configGroupName = QString::null;
   this->autoUpdate = false;

   this->modified = false;
   this->unmodifiedValue = value;

   this->saveToGroup = group;
   this->group = group;

}


template <class T>
coConfigValue<T>::coConfigValue(coConfigGroup * group,
                                const QString & variable, const QString & section)
{

   this->variable = variable;
   this->section = section;
   this->configGroupName = QString::null;
   this->autoUpdate = false;

   this->modified = false;
   this->unmodifiedValue = value;

   this->saveToGroup = group;
   this->group = group;

}


template <class T>
coConfigValue<T>::coConfigValue(const coConfigValue<T> & value)
{

   //COCONFIGLOG("coConfigValue<T>::<init> info: copy");

   this->value = value.value;
   this->variable = value.variable;
   this->section = value.section;
   this->configGroupName = value.configGroupName;

   this->autoUpdate = value.autoUpdate;

   this->modified = value.modified;
   this->unmodifiedValue = value.unmodifiedValue;

   this->saveToGroup = value.saveToGroup;
   this->group = value.group;

#ifdef COCONFIGVALUE_USE_CACHE
   this->cache = value.cache;
#endif
}


template <class T>
coConfigValue<T>::~coConfigValue()
{

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
   else if(!configGroupName.isNull())
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
      value = group->getValue(variable, section);
      //std::cerr << "coConfigValue<T>::update info: group value " << section << "." << variable << " = " << (value.isNull() ? "*NULL*" : value) << std::endl;
   }
   else
   {
      value = coConfig::getInstance()->getValue(variable, section);
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
   if (isAutoUpdate()) update();
   //COCONFIGLOG("coConfigValue<T>::hasValidValue info: value " << section << "." << variable << " = " << (value.isNull() ? "*NULL*" : value));
   return !value.isNull();
}


template <class T>
bool coConfigValue<T>::hasValidValue() const
{
   //COCONFIGLOG("coConfigValue<T>::hasValidValue info: value " << section << "." << variable << " = " << (value.isNull() ? "*NULL*" : value));
   return !value.isNull();
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
