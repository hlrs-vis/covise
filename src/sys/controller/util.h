/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROL_UTIL_H
#define CONTROL_UTIL_H

#include <string>
#include <vector>

namespace covise{
namespace controller
{
    class CTRLHandler;
    class Userinterface;
    std::string coviseBinDir();
    
    //comments are substrings starting with #
    std::vector<std::string> splitStringAndRemoveComments(const std::string &str, const std::string &sep);
    bool load_config(const std::string &filename, CTRLHandler &handler, const std::vector<const Userinterface *> &uis);

} // namespace controller
}// namespace covise

#endif // !CONTROL_UTIL_H