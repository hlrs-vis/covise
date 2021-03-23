/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "util.h"
#include "userinterface.h"
#include "handler.h"

#include <net/message.h>
#include <net/message_types.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>

using namespace covise;
using namespace covise::controller;

std::string covise::controller::coviseBinDir()
{
    static std::string binDir;
    if (binDir.empty())
    {

        const char *covisedir = getenv("COVISEDIR");
        const char *archsuffix = getenv("ARCHSUFFIX");
        std::stringstream ss;

        if (covisedir && archsuffix)
        {
            ss << covisedir << "/" << archsuffix << "/bin/";
            binDir = ss.str();
        }
    }
    return binDir;
}

std::vector<std::string> covise::controller::splitStringAndRemoveComments(const std::string &str, const std::string &sep)
{
    std::vector<std::string> list;
    boost::split(list, str, boost::is_any_of(sep));
    list.erase(std::remove_if(list.begin(), list.end(), [](const std::string &s) {
                   return s[0] == '#';
               }),
               list.end());
    while (list.size() > 0 && list.back() == "")
    {
        list.pop_back();
    }

    return list;
}

constexpr size_t NET_FILE_VERSION = 632;

std::string openNetFile(const std::string &filename, const std::vector<const Userinterface *> uis)
{
    std::filebuf *pbuf = NULL;

    // try to read given file or the UNDO>NET file
    // otherwise send an error message

    std::ifstream inFile(filename.c_str());
    if (!inFile.good())
    {
        std::string covisepath = getenv("COVISEDIR");
        if (filename == "UNDO.NET" && !covisepath.empty())
        {
            std::string filestr = covisepath + "/" + filename;
            std::ifstream inFile2(filestr.c_str(), std::ifstream::in);
            if (!inFile2.good())
            {
                Message err_msg{COVISE_MESSAGE_COVISE_ERROR, "Can't open file " + filename};
                for (const auto &ui : uis)
                    ui->send(&err_msg);
            }
            else
                pbuf = inFile2.rdbuf();
        }
    }
    else
        pbuf = inFile.rdbuf();

    if (pbuf)
    {
        try
        {
            std::streamsize size = pbuf->pubseekoff(0, ios::end, ios::in);
            pbuf->pubseekpos(0, ios::in);
            std::string buffer;
            buffer.resize(size + 1);
            std::streamsize got = pbuf->sgetn(&buffer[0], size);
            buffer[got] = '\0';
            return buffer;
        }
        catch (const std::exception &)
        {
            std::cerr << "net_module_list::openNetFile: error reading " << filename << std::endl;
        }
    }
    return std::string{};
}

bool covise::controller::load_config(const std::string &filename, CTRLHandler &handler, const std::vector<const Userinterface *> &uis)
{
    auto buffer = openNetFile(filename, uis);
    // read content

    if (!buffer.empty())
    {
        bool oldFile = true;
        if (buffer[0] == '#')
        {
            //we have a version information
            int version;
            int n = sscanf(buffer.c_str() + 1, "%d", &version);
            if (n == 1 && version >= NET_FILE_VERSION)
            {
                oldFile = false; // this is a new .net file, convert all other files
            }
        }

        if (oldFile)
        {
            // convert
            string path = filename;
#ifdef WIN32
            for (int i = 0; i < path.length(); i++)
            {
                if (path[i] == '/')
                    path[i] = '\\';
            }
            string name = path;
            for (int i = (int)path.length() - 1; i >= 0; i--)
            {
                if (path[i] == '\\')
                {
                    name = string(path, i + 1, path.length() - i);
                    break;
                }
            }

            std::string command = "map_converter -f -o " + path + ".new " + path;
#else
            std::string command = "map_converter -f -o \"" + path + ".new\" \"" + path + "\"";
#endif
            if (system(command.c_str()) == 0)
            {
#ifdef WIN32
                command = "rename " + path + " " + name + ".bak";
#else
                command = "mv \"" + path + "\" \"" + path + ".bak\"";
#endif
                if (system(command.c_str()) == 0)
                {
#ifdef WIN32
                    command = "rename " + path + ".new " + name;
#else
                    command = "mv \"" + path + ".new\" \"" + path + "\"";
#endif
                    if (system(command.c_str()) == 0)
                    {
                        // read again
                        buffer = openNetFile(filename, uis);
                        oldFile = false;

                        Message tmpmsg{COVISE_MESSAGE_UI, "CONVERTED_NET_FILE\n" + filename};
                        for (const auto &ui : uis)
                            ui->send(&tmpmsg);
                    }
                }
            }
        }

        if (oldFile)
        {
            // conversion failed
            Message tmpmsg{COVISE_MESSAGE_UI, "FAILED_NET_FILE_CONVERSION\n" + filename};
            for (const auto &ui : uis)
                ui->send(&tmpmsg);
        }
        else
        {
            return handler.recreate(buffer, CTRLHandler::NETWORKMAP);
        }
    }
    return false;
}