/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGEDITORIMPORT_H
#define COCONFIGEDITORIMPORT_H

#include <config/coConfigConstants.h>
#include <config/coConfigImportReader.h>

#include "coQtFileSelection.h"

#include <qstring.h>
#include <util/coTypes.h>

class Q3Wizard;

class CONFIGEDITOREXPORT coConfigEditorImport : public QObject
{

    Q_OBJECT

public:
    coConfigEditorImport(const QString &source = QString::null,
                         const QString &dest = QString::null,
                         const QString &transform = QString::null);

    ~coConfigEditorImport();

    QString getTransformInstructionFile() const;
    QString getSourceFile() const;
    QString getDestinationFile() const;

public slots:
    void setTransformInstructionFile(const QString &filename);
    void setSourceFile(const QString &filename);
    void setDestinationFile(const QString &filename);

    void importWizard();

private:
    coQtFileSelection *makeFileSelectionWidget();

private:
    QString transformFilename;
    QString sourceFilename;
    QString destinationFilename;

    Q3Wizard *wizard;
};

#endif
