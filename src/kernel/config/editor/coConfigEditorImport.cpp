/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coConfigEditorImport.h"
//#include "coConfigEditorImport.moc"

#include <sys/types.h>
#include <sys/stat.h>

#include <qdir.h>
#include <qdom.h>
#include <qlabel.h>
#include <q3wizard.h>

coConfigEditorImport::coConfigEditorImport(const QString &source,
                                           const QString &dest,
                                           const QString &transform)
    : QObject()
{

    setTransformInstructionFile(transform);
    setDestinationFile(dest);
    setSourceFile(source);

    wizard = 0;
}

coConfigEditorImport::~coConfigEditorImport() {}

void coConfigEditorImport::setTransformInstructionFile(const QString &filename)
{
    //cerr << "coConfigEditorImport::setTransformInstructionFile info: set "
    //     << filename << endl;
    transformFilename = filename;
}

QString coConfigEditorImport::getTransformInstructionFile() const
{
    return transformFilename;
}

void coConfigEditorImport::setSourceFile(const QString &filename)
{
    //cerr << "coConfigEditorImport::setSourceFile info: set "
    //     << filename << endl;
    sourceFilename = filename;
}

QString coConfigEditorImport::getSourceFile() const
{
    return sourceFilename;
}

void coConfigEditorImport::setDestinationFile(const QString &filename)
{
    //cerr << "coConfigEditorImport::setDestinationFile info: set "
    //     << filename << endl;
    destinationFilename = filename;
}

QString coConfigEditorImport::getDestinationFile() const
{
    return destinationFilename;
}

void coConfigEditorImport::importWizard()
{

    delete wizard;
    wizard = new Q3Wizard(0, 0, true);

    coQtFileSelection *source = makeFileSelectionWidget();
    coQtFileSelection *dest = makeFileSelectionWidget();
    coQtFileSelection *trans = makeFileSelectionWidget();

    connect(source, SIGNAL(fileNameChanged(const QString &)),
            this, SLOT(setSourceFile(const QString &)));

    connect(dest, SIGNAL(fileNameChanged(const QString &)),
            this, SLOT(setDestinationFile(const QString &)));

    connect(trans, SIGNAL(fileNameChanged(const QString &)),
            this, SLOT(setTransformInstructionFile(const QString &)));

//source->setFileName("yac.config.in");
#ifdef YAC
    char *covisepath = getenv("YACDIR");
#else
    char *covisepath = getenv("COVISEDIR");
#endif
    source->setFileName(QString(covisepath) + "/config/covise.config");
    dest->setFileName(QDir::homeDirPath() + "/.covise/config-test.xml");
    //trans->setFileName(coConfigDefaultPaths::getDefaultTransformFileName());
    trans->setFileName(QString(covisepath) + "/config/transform.xml");

    wizard->addPage(source, wizard->tr("Source Filename"));
    wizard->addPage(dest, wizard->tr("Destination Filename"));
    wizard->addPage(trans, wizard->tr("Transformation Filename"));

    wizard->setHelpEnabled(source, false);
    wizard->setHelpEnabled(dest, false);
    wizard->setHelpEnabled(trans, false);

    wizard->setFinishEnabled(trans, true);

    if (wizard->exec() == Q3Wizard::Accepted)
    {

        coConfigImportReader *reader = new coConfigImportReader(sourceFilename,
                                                                destinationFilename,
                                                                transformFilename);

        reader->parse();
        reader->write();
    }
}

coQtFileSelection *coConfigEditorImport::makeFileSelectionWidget()
{

    coQtFileSelection *fs = new coQtFileSelection(QString::null, wizard);
    return fs;
}
