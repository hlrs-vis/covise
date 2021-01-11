/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <algorithm>
#include <array>

#include "moduleInfo.h"
#include "exception.h"
#include "util.h"

using namespace covise;
using namespace covise::controller;

void ModuleNetConnectivity::addInterface(const std::string &name, const std::string &type, controller::Direction direction, const std::string &text, const std::string &demand)
{
    auto inter = interfaces.emplace(interfaces.end(), new C_interface{});
    inter->get()->set_name(name);
    inter->get()->set_type(type);
    inter->get()->set_direction(direction);
    inter->get()->set_text(text);
    inter->get()->set_demand(demand);
}

void ModuleNetConnectivity::addParameter(const std::string &name, const std::string &type, const std::string &text, const std::string &value, const std::string ext, Direction dir)
{
    parameter *param;
    if (dir == Direction::Input)
        param = &*inputParams.emplace(inputParams.end());
    else if (dir == Direction::Output)
        param = &*outputParams.emplace(outputParams.end());

    param->set_name(name);
    param->set_type(type);
    param->set_text(text);
    param->set_extension(ext);
    param->set_value_list(value);
}

const parameter &ModuleNetConnectivity::getParam(const std::string &paramName) const
{
    return const_cast<ModuleNetConnectivity *>(this)->getParam(paramName);
}

parameter &ModuleNetConnectivity::getParam(const std::string &paramName)
{
    auto param = getParameter(inputParams, paramName);
    if (!param)
    {
        param = getParameter(outputParams, paramName);
    }
    if (!param)
    {
        throw Exception{"ModuleNetConnectivity did not find parameter " + paramName};
    }
    return *param;
}

void ModuleNetConnectivity::forAllNetInterfaces(const std::function<void(net_interface &)> &func)
{
    for (auto &inter : interfaces)
    {
        if (auto netInter = dynamic_cast<net_interface *>(inter.get()))
            func(*netInter);
    }
}

void ModuleNetConnectivity::forAllNetInterfaces(const std::function<void(const net_interface &)> &func) const
{
    for (auto &inter : interfaces)
    {
        if (auto netInter = dynamic_cast<const net_interface *>(inter.get()))
            func(*netInter);
    }
}

const parameter *controller::getParameter(const std::vector<parameter> &params, const std::string &parameterName)
{
    auto it = std::find_if(params.begin(), params.end(), [&parameterName](const parameter &param) {
        return param.get_name() == parameterName;
    });
    if (it != params.end())
    {
        return &*it;
    }
    return nullptr;
}

parameter *controller::getParameter(std::vector<parameter> &params, const std::string &parameterName)
{
    return const_cast<parameter *>(getParameter(const_cast<const std::vector<parameter> &>(params), parameterName));
}

ModuleInfo::ModuleInfo(const std::string &name, const std::string &category)
    : name(name), category(category) {}

const ModuleNetConnectivity &ModuleInfo::connectivity() const
{
    return m_connectivity;
}

const std::string &ModuleInfo::description() const
{
    return m_description;
}

void ModuleInfo::readConnectivity(const char *data) const
{
    m_connectivity.interfaces.clear();
    m_connectivity.inputParams.clear();
    m_connectivity.outputParams.clear();
    auto list = splitStringAndRemoveComments(data, "\n");
    int iel = 3;
    m_description = list[iel++];
    std::array<int, 4> interfaceAndParamCounts; //in_interface, out_interface, in_param, out_param
    for (size_t i = 0; i < 4; i++)
    {
        interfaceAndParamCounts[i] = std::stoi(list[iel++]);
    }
    for (size_t i = 0; i < interfaceAndParamCounts[0]; i++) // read the input -interfaces
    {
        m_connectivity.addInterface(list[iel], list[iel + 1], Direction::Input, list[iel + 2], list[iel + 3]);
        iel += 4;
    }
    for (size_t i = 0; i < interfaceAndParamCounts[1]; i++) // read the output -interfaces
    {
        m_connectivity.addInterface(list[iel], list[iel + 1], Direction::Output, list[iel + 2], list[iel + 3]);
        iel += 4;
    }
    for (size_t i = 0; i < interfaceAndParamCounts[2]; i++) // read the output -interfaces
    {
        m_connectivity.addParameter(list[iel], list[iel + 1], list[iel + 2], list[iel + 3], list[iel + 4], Direction::Input);
        iel += 5;
    }
    for (size_t i = 0; i < interfaceAndParamCounts[3]; i++) // read the output -interfaces
    {
        m_connectivity.addParameter(list[iel], list[iel + 1], list[iel + 2], list[iel + 3], list[iel + 4], Direction::Output);
        iel += 5;
    }
}

bool ModuleInfo::operator==(const ModuleInfo &other) const
{
    return name == other.name;
}

bool ModuleInfo::operator<(const ModuleInfo &other) const
{
    return name < other.name;
}
