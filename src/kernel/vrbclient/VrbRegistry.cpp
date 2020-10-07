/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VrbRegistry.h"
namespace vrb
{

const regClass* VrbRegistry::getClass(const std::string& name) const
{
	return const_cast<VrbRegistry *>(this)->getClass(name);
}

regClass* VrbRegistry::getClass(const std::string& name)
{
	auto cl = findClass(name);
	if (cl == end())
	{
		return nullptr;
	}
	return cl->get();
}


void VrbRegistry::deserialize(covise::TokenBuffer& tb) {

	size_t size;
	tb >> size;
	m_classes = ContainerType{size, createClass("", -1)};// -1 = nobodies client ID
	for(auto &cl : m_classes)
	{
		vrb::deserialize(tb, *cl);
	}
}
void VrbRegistry::serialize(covise::TokenBuffer& tb) const{
	tb << m_classes.size();
	for (const auto &cl : m_classes)
	{
		vrb::serialize(tb, *cl);
	}
}

VrbRegistry::ContainerType::iterator VrbRegistry::begin() {
	return m_classes.begin();
}
VrbRegistry::ContainerType::const_iterator VrbRegistry::begin() const {
	return m_classes.begin();
}
VrbRegistry::ContainerType::iterator VrbRegistry::end() {
	return m_classes.end();
}
VrbRegistry::ContainerType::const_iterator VrbRegistry::end() const {
	return m_classes.end();
}

VrbRegistry::ContainerType::iterator VrbRegistry::findClass(const std::string &className){
return std::find_if(begin(), end(), [className](std::shared_ptr<regClass> cl) { return cl->name() == className; });
}

}

