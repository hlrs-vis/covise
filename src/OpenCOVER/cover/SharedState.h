/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! Basic class to synchronize the state of any variable with other clients connectet to vrb
for variables of types other than: bool, int, float, double, string, TokenBuffer
the serialize and deserialize methods have to be implemented
make sure the variable name is unique for each SharedState e.g. by naming the variable Plugin.Class.Variablename.ID

*/
#ifndef VRB_SHAREDSTATE_H
#define VRB_SHAREDSTATE_H
	
#include <string>

#include <net/tokenbuffer.h>
#include <net/message.h>
#include <net/message_types.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVrbRegistryAccess.h>
#include <cover/coVRCommunication.h>


namespace opencover {

///convert the value to a TokenBuffer
template<class T>
void serialize(covise::TokenBuffer &tb, const T &value)
{
	tb << value;
}
///converts the TokenBuffer back to the value
template<class T>
void deserialize(covise::TokenBuffer &tb, T &value)
{
	tb >> value;
}
template <>
void serialize<std::vector<std::string>>(covise::TokenBuffer &tb, const std::vector<std::string> &value);

template <>
void deserialize<std::vector<std::string>>(covise::TokenBuffer &tb, std::vector<std::string> &value);

template <class T>
class  SharedState : public coVrbRegEntryObserver
{
public:
	SharedState<T>(std::string name, T value = T())
		:variableName(name)
		,m_value(value)
	{
		m_registry = coVRCommunication::instance()->registry;
        covise::TokenBuffer data;
        serialize(data, m_value);
        regEntry = m_registry->subscribeVar(className.c_str(), 8, name.c_str(), std::move(data), this); //ID != 0;

	}
    SharedState<T> &operator=(T value)
    {
        if (m_value != value)
        {
            m_value = value;
            push();
        }
        return *this;
    }
    operator T() const
    {
        return m_value;
    }
    ///sends the value change to the vrb
    void push()
    {
        valueChanged = false;
        covise::TokenBuffer data;
        serialize(data, m_value);
        regEntry->setData(std::move(data));
    }
    //is called from the registryAcces when the registry entry got changed from the server
	void update(coVrbRegEntry *theChangedRegEntry) override
    {
        if (strcmp(theChangedRegEntry->getVar(), variableName.c_str()))
        {
            return;
        }
        theChangedRegEntry->getData().rewind();
		deserialize (theChangedRegEntry->getData(), m_value);
		valueChanged = true;
		if (updateCallback != nullptr)
		{
			updateCallback();
		}
	}
	
	//returns true if the last value change was made by an other client
	bool valueChangedByOther() {
		return valueChanged;
	}
	std::string getName() {
		return variableName;
	}
	///let the SharedState call the given function when the registry entry got changed from the server
    void setUpdateFunction(std::function<void(void)> function ) {
		updateCallback = function;
	}
    T &value()
    {
        return m_value;
    }
private:
    const std::string className = "SharedState";
    std::string variableName;
    T m_value;
    coVrbRegistryAccess *m_registry;
    coVrbRegEntry *regEntry = nullptr;
	bool doSend = false;
	bool doReceive = false;
	bool valueChanged = false;
    std::function<void(void)> updateCallback = nullptr;
};	
}
#endif
