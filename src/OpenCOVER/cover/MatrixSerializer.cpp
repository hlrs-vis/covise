/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

   * License: LGPL 2+ */

#include "MatrixSerializer.h"

namespace vrb {
template <>
void serialize<osg::Matrix>(covise::TokenBuffer &tb, const osg::Matrix &value) {

    tb << value(0, 0);  tb << value(0, 1); tb << value(0, 2); tb << value(0, 3);
    tb << value(1, 0);  tb << value(1, 1); tb << value(1, 2); tb << value(1, 3);
    tb << value(2, 0);  tb << value(2, 1); tb << value(2, 2); tb << value(2, 3);
    tb << value(3, 0);  tb << value(3, 1); tb << value(3, 2); tb << value(3, 3);

}
template<>
void deserialize<osg::Matrix>(covise::TokenBuffer &tb, osg::Matrix &value) {
    tb >> value(0, 0); tb >> value(0, 1); tb >> value(0, 2); tb >> value(0, 3);
    tb >> value(1, 0); tb >> value(1, 1); tb >> value(1, 2); tb >> value(1, 3);
    tb >> value(2, 0); tb >> value(2, 1); tb >> value(2, 2); tb >> value(2, 3);
    tb >> value(3, 0); tb >> value(3, 1); tb >> value(3, 2); tb >> value(3, 3);

}
}


