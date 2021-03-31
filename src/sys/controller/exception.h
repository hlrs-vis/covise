/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROLL_EXCECPTION_H
#define CONTROLL_EXCECPTION_H
#include <stdexcept>
#include <string>
namespace covise{
namespace controller
{
struct Exception : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

} // namespace controller
} // namespace covise

#endif // !CONTROLL_EXECPTION_H