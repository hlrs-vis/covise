/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_CLIP_PLANE_EDITOR_
#define _INV_CLIP_PLANE_EDITOR_

#include <util/coTypes.h>

#include <QDialog>

class QLineEdit;

//============================================================================
//
//  Class: InvClipPlaneEditor
//
//  This editor lets you interactively set a clipping plane.
//
//============================================================================
class InvClipPlaneEditor : public QDialog
{

    Q_OBJECT

public:
    InvClipPlaneEditor(QWidget *parent = 0, const char *name = "addhost");
    ~InvClipPlaneEditor();

private slots:
    void equationCB();

private:
    QLineEdit *p[3], *n[3];
    double pt[3], nl[3];
};
#endif // _INV_CLIP_PLANE_EDITOR_
