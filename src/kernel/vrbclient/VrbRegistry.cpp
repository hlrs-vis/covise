/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VrbRegistry.h"
#include "regClass.h"
namespace vrb
{
void VrbRegistry::readClass(std::ifstream& file)
{
	std::string className, delimiter, space;
	file >> className;
	file >> space;
	std::shared_ptr<regClass> cl = createClass(className, -1); // -1 = nobodies client ID
	myClasses[className] = cl;
	cl->readVar(file);
}

regClass* VrbRegistry::getClass(const std::string& name) const
{

	auto cl = myClasses.find(name);
	if (cl == myClasses.end())
	{
		return nullptr;
	}
	return cl->second.get();
}

void VrbRegistry::loadRegistry(std::ifstream& inFile) {

	while (!inFile.eof())
	{
		readClass(inFile);
	}

}

void VrbRegistry::saveRegistry(std::ofstream& outFile) const {
	for (const auto& cl : myClasses)
	{
		outFile << "\n";
		cl.second->writeClass(outFile);
	}
}
}
