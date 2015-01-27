/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include "MESpinBox.h"
#include "ports/MEParameterPort.h"

/*****************************************************************************
 *
 * Class MESpinBox
 *
 *****************************************************************************/

MESpinBox::MESpinBox(MEParameterPort *p, QWidget *parent)
    : QSpinBox(parent)
    , port(p)
{
}

QString MESpinBox::mapValueToText(int ival)
{
    Q_UNUSED(ival);
    //   float val = port->fmin +  ( float( ival) / 1000.* ( port->fmax - port->fmin ) );
    //   return QString::number(val);
    return "NO";
}

int MESpinBox::mapTextToValue(bool *)
{
    //  float inp =  text().toFloat();
    //   int ival = int(inp * 1000) / int( port->fmax - port->fmin );
    //  return ival;
    return 0;
}
