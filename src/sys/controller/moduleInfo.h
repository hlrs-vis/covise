/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef CONTROLLER_MODULE_INFO_H
#define CONTROLLER_MODULE_INFO_H

#include "port.h"
#include "exception.h"
#include <string>
#include <vector>

namespace covise
{
    namespace controller
    {
        //Holds the input and output ports of a module and their parameters
        struct ModuleNetConnectivity
        {
            ModuleNetConnectivity() = default;
            ModuleNetConnectivity(const ModuleNetConnectivity &other) = delete;
            ModuleNetConnectivity(ModuleNetConnectivity &&other) = default;
            ModuleNetConnectivity &operator=(const ModuleNetConnectivity &other) = delete;
            ModuleNetConnectivity &operator=(ModuleNetConnectivity &&other) = default;
            ~ModuleNetConnectivity() = default;

            std::vector<std::unique_ptr<C_interface>> interfaces;
            std::vector<parameter> inputParams, outputParams;
            void addInterface(const std::string &name, const std::string &type, Direction direction, const std::string &text, const std::string &demand);
            void addParameter(const std::string &name, const std::string &type, const std::string &text, const std::string &value, const std::string ext, Direction dir);

            const parameter &getParam(const std::string &paramName) const;
            parameter &getParam(const std::string &paramName);

            template <typename T>
            T &getInterface(const std::string &interfaceName)
            {
                auto it = std::find_if(interfaces.begin(), interfaces.end(), [&interfaceName](const std::unique_ptr<C_interface> &inter) {
                    return inter->get_name() == interfaceName;
                });
                if (it == interfaces.end())
                {
                    throw Exception{"ModuleNetConnectivity did not find interface " + interfaceName};
                }
                if (auto netInter = dynamic_cast<T *>(it->get()))
                    return *netInter;
                throw Exception{"ModuleNetConnectivity did not find interface: type mismatch"};
            }
            template <typename T>
            const T &getInterface(const std::string &interfaceName) const
            {
                return const_cast<ModuleNetConnectivity *>(this)->getInterface<T>(interfaceName);
            }

            void forAllNetInterfaces(const std::function<void(net_interface &)> &func);
            void forAllNetInterfaces(const std::function<void(const net_interface &)> &func) const;
        };

        const parameter *getParameter(const std::vector<parameter> &params, const std::string &parameterName);
        parameter *getParameter(std::vector<parameter> &params, const std::string &parameterName);

        //name, category and connectivity info of a module class
        //and the module count wich is used to give each instance of a module its instance number
        struct ModuleInfo
        {
            ModuleInfo(const std::string &name, const std::string &category);
            const std::string name;     //executable of the module must be under $COVISE_PATH/$ARCHSUFFIX/bin/name
            const std::string category; //if set used as sub-directory in $COVISE_PATH/$ARCHSUFFIX/bin/category/name
            mutable size_t count = 0;
            const ModuleNetConnectivity &connectivity() const;
            const std::string &description() const;
            void readConnectivity(const char *buff) const;
            bool operator==(const ModuleInfo &other) const;
            bool operator<(const ModuleInfo &other) const;

        private:
            mutable std::string m_description;
            mutable ModuleNetConnectivity m_connectivity;
        };

    } // namespace controller
} // namespace covise
#endif // !CONTROLLER_MODULE_INFO_H
