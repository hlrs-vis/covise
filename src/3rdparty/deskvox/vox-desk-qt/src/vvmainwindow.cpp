// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include "vvcanvas.h"
#include "vvclipdialog.h"
#include "vvdimensiondialog.h"
#include "vvlightdialog.h"
#include "vvmainwindow.h"
#include "vvmergedialog.h"
#include "vvobjview.h"
#include "vvplugin.h"
#include "vvpluginutil.h"
#include "vvprefdialog.h"
#include "vvscreenshotdialog.h"
#include "vvshortcutdialog.h"
#include "vvsliceviewer.h"
#include "vvtfdialog.h"
#include "vvtimestepdialog.h"
#include "vvvolinfodialog.h"

#include "ui_vvmainwindow.h"

#include <virvo/fileio/feature.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvmacros.h>
#include <virvo/vvvoldesc.h>

#include <QApplication>
#include <QByteArray>
#include <QColorDialog>
#include <QCoreApplication>
#include <QFileDialog>
#include <QFileInfo>
#include <QList>
#include <QMessageBox>
#include <QSettings>
#include <QShortcut>
#include <QStringList>

using vox::vvObjView;

struct vvMainWindow::Impl
{
  Impl() : ui(new Ui::MainWindow) {}

  std::auto_ptr<Ui::MainWindow> ui;

  QList<vvPlugin*> plugins;

  // pointers managed by main-window object
  vvCanvas* canvas;
  vvClipDialog* clipDialog;
  vvPrefDialog* prefDialog;
  vvDimensionDialog* dimensionDialog;
  vvMergeDialog* mergeDialog;
  vvScreenshotDialog* screenshotDialog;
  vvShortcutDialog* shortcutDialog;
  vvTFDialog* tfDialog;
  vvLightDialog* lightDialog;
  vvSliceViewer* sliceViewer;
  vvTimeStepDialog* timeStepDialog;
  vvVolInfoDialog* volInfoDialog;

private:

  VV_NOT_COPYABLE(Impl)

};

namespace
{
std::vector<std::string> getRecentFiles()
{
  QSettings settings;
  QString qrecent = settings.value("canvas/recentfiles").toString();
  std::string recent = qrecent.toStdString();
  std::vector<std::string> files = vvToolshed::split(recent, ";");
  return files;
}

void addRecentFile(const QString& filename)
{
  std::vector<std::string> files = getRecentFiles();
  if (std::find(files.begin(), files.end(), filename.toStdString()) == files.end())
  {
    files.push_back(filename.toStdString());
  }

  const static size_t BUFSIZE = 10;
  size_t first = files.size() < BUFSIZE ? 0 : files.size() - BUFSIZE;

  QString recentstr;
  for (size_t i = first; i < files.size(); ++i)
  {
    recentstr.append(files[i].c_str());
    if (i < files.size() - 1)
    {
      recentstr.append(";");
    }
  }
  QSettings settings;
  settings.setValue("canvas/recentfiles", recentstr);
}
}

vvMainWindow::vvMainWindow(const QString& filename, QWidget* parent)
  : QMainWindow(parent)
  , impl_(new Impl)
{
  vvDebugMsg::msg(1, "vvMainWindow::vvMainWindow()");

  impl_->ui->setupUi(this);

  // plugins
  impl_->plugins = vvPluginUtil::getAll();
  foreach (vvPlugin* plugin, impl_->plugins)
  {
    if (QDialog* dialog = plugin->dialog(this))
    {
      impl_->ui->menuPlugins->setEnabled(true);
      QAction* dialogAction = new QAction(plugin->name(), impl_->ui->menuPlugins);
      impl_->ui->menuPlugins->addAction(dialogAction);
      connect(dialogAction, SIGNAL(triggered()), dialog, SLOT(show()));
    }
  }

  // widgets and dialogs
  const int superSamples = 0;
  QGLFormat format;
  format.setDoubleBuffer(true);
  format.setDepth(true);
  format.setRgba(true);
  format.setAlpha(true);
  format.setAccum(true);
  format.setStencil(true); // needed for interlaced stereo
  if (superSamples > 0)
  {
    format.setSampleBuffers(true);
    format.setSamples(superSamples);
  }

  QString fn = "";
  if (filename != "")
  {
    QFileInfo finfo(filename);
    if (finfo.exists())
    {
      if (finfo.isFile()) // TODO: we really need file type validation in virvo::FileIO w/o having to load the actual data
      {
        fn = filename;
      }
      else
      {
        QMessageBox::warning(this, tr("Error loading file"), tr("No file or link to a file: ") + filename, QMessageBox::Ok);
      }
    }
    else
    {
      QMessageBox::warning(this, tr("Error loading file"), tr("File not found: ") + filename, QMessageBox::Ok);
    }
  }

  impl_->canvas = new vvCanvas(format, fn, this);
  impl_->canvas->setPlugins(impl_->plugins);
  setCentralWidget(impl_->canvas);

  impl_->ui->menuRecentVolumes->clear();
  std::vector<std::string> recents = getRecentFiles();
  for (std::vector<std::string>::const_iterator it = recents.begin();
       it != recents.end(); ++it)
  {
    QAction* action = new QAction((*it).c_str(), this);
    connect(action, SIGNAL(triggered()), this, SLOT(onRecentVolumeTriggered()));
    impl_->ui->menuRecentVolumes->insertAction(*impl_->ui->menuRecentVolumes->actions().begin(), action);
  }

  if (fn != "")
  {
    addRecentFile(fn);
  }

  impl_->prefDialog = NULL;

  impl_->tfDialog = new vvTFDialog(impl_->canvas, this);
  impl_->clipDialog = new vvClipDialog(this);
  impl_->lightDialog = new vvLightDialog(this);

  impl_->dimensionDialog = new vvDimensionDialog(impl_->canvas, this);
  impl_->mergeDialog = new vvMergeDialog(this);
  impl_->screenshotDialog = new vvScreenshotDialog(impl_->canvas, this);
  impl_->shortcutDialog = new vvShortcutDialog(this);
  impl_->sliceViewer = new vvSliceViewer(impl_->canvas->getVolDesc(), this);
  impl_->timeStepDialog = new vvTimeStepDialog(this);
  impl_->volInfoDialog = new vvVolInfoDialog(this);

  // file menu
  connect(impl_->ui->actionLoadVolume, SIGNAL(triggered()), this, SLOT(onLoadVolumeTriggered()));
  connect(impl_->ui->actionReloadVolume, SIGNAL(triggered()), this, SLOT(onReloadVolumeTriggered()));
  connect(impl_->ui->actionSaveVolumeAs, SIGNAL(triggered()), this, SLOT(onSaveVolumeAsTriggered()));
  connect(impl_->ui->actionMergeFiles, SIGNAL(triggered()), this, SLOT(onMergeFilesTriggered()));
  connect(impl_->ui->actionLoadCamera, SIGNAL(triggered()), this, SLOT(onLoadCameraTriggered()));
  connect(impl_->ui->actionSaveCameraAs, SIGNAL(triggered()), this, SLOT(onSaveCameraAsTriggered()));
  connect(impl_->ui->actionScreenshot, SIGNAL(triggered()), this, SLOT(onScreenshotTriggered()));
  connect(impl_->ui->actionPreferences, SIGNAL(triggered()), this, SLOT(onPreferencesTriggered()));

  // settings menu
  connect(impl_->ui->actionTransferFunction, SIGNAL(triggered()), this, SLOT(onTransferFunctionTriggered()));
  connect(impl_->ui->actionClippingPlane, SIGNAL(triggered()), this, SLOT(onClippingPlaneTriggered()));
  connect(impl_->ui->actionLightSource, SIGNAL(triggered()), this, SLOT(onLightSourceTriggered()));
  connect(impl_->ui->actionBackgroundColor, SIGNAL(triggered()), this, SLOT(onBackgroundColorTriggered()));

  // edit menu
  connect(impl_->ui->actionSampleDistances, SIGNAL(triggered()), this, SLOT(onSampleDistancesTriggered()));

  // view menu
  connect(impl_->ui->actionShowOrientation, SIGNAL(triggered(bool)), this, SLOT(onShowOrientationTriggered(bool)));
  connect(impl_->ui->actionShowBoundaries, SIGNAL(triggered(bool)), this, SLOT(onShowBoundariesTriggered(bool)));
  connect(impl_->ui->actionShowPalette, SIGNAL(triggered(bool)), this, SLOT(onShowPaletteTriggered(bool)));
  connect(impl_->ui->actionShowNumTextures, SIGNAL(triggered(bool)), this, SLOT(onShowNumTexturesTriggered(bool)));
  connect(impl_->ui->actionShowFrameRate, SIGNAL(triggered(bool)), this, SLOT(onShowFrameRateTriggered(bool)));
  connect(impl_->ui->actionAutoRotation, SIGNAL(triggered(bool)), this, SLOT(onAutoRotationTriggered(bool)));
  connect(impl_->ui->actionVolumeInformation, SIGNAL(triggered(bool)), this, SLOT(onVolumeInformationTriggered()));
  connect(impl_->ui->actionSliceViewer, SIGNAL(triggered()), this, SLOT(onSliceViewerTriggered()));
  connect(impl_->ui->actionTimeSteps, SIGNAL(triggered()), this, SLOT(onTimeStepsTriggered()));

  // help menu
  connect(impl_->ui->actionKeyboardCommands, SIGNAL(triggered()), this, SLOT(onKeyboardCommandsClicked()));

  // misc.
  connect(impl_->canvas, SIGNAL(newVolDesc(vvVolDesc*)), this, SLOT(onNewVolDesc(vvVolDesc*)));
  connect(impl_->canvas, SIGNAL(statusMessage(const std::string&)), this, SLOT(onStatusMessage(const std::string&)));

  connect(impl_->tfDialog, SIGNAL(newWidget(vvTFWidget*)), impl_->canvas, SLOT(addTFWidget(vvTFWidget*)));
  connect(impl_->tfDialog, SIGNAL(newTransferFunction()), impl_->canvas, SLOT(updateTransferFunction()));
  connect(impl_->tfDialog, SIGNAL(undo()), impl_->canvas, SLOT(undoTransferFunction()));
  connect(impl_->tfDialog, SIGNAL(newTransferFunction()), impl_->sliceViewer, SLOT(update()));

  connect(impl_->clipDialog, SIGNAL(clipping(bool)), impl_->canvas, SLOT(enableClipping(bool)));
  connect(impl_->clipDialog, SIGNAL(normal(virvo::vec3f const&)), impl_->canvas, SLOT(setClipNormal(virvo::vec3f const&)));
  connect(impl_->clipDialog, SIGNAL(origin(virvo::vec3f const&)), impl_->canvas, SLOT(setClipOrigin(virvo::vec3f const&)));
  connect(impl_->clipDialog, SIGNAL(singleSlice(bool)), impl_->canvas, SLOT(setClipSingleSlice(bool)));
  connect(impl_->clipDialog, SIGNAL(opaque(bool)), impl_->canvas, SLOT(setClipOpaque(bool)));
  connect(impl_->clipDialog, SIGNAL(perimeter(bool)), impl_->canvas, SLOT(setClipPerimeter(bool)));

  connect(impl_->lightDialog, SIGNAL(enabled(bool)), impl_->canvas, SLOT(enableLighting(bool)));
  connect(impl_->lightDialog, SIGNAL(showLightSource(bool)), impl_->canvas, SLOT(showLightSource(bool)));
  connect(impl_->lightDialog, SIGNAL(enableHeadlight(bool)), impl_->canvas, SLOT(enableHeadlight(bool)));
  connect(impl_->lightDialog, SIGNAL(editPositionToggled(bool)), impl_->canvas, SLOT(editLightPosition(bool)));
  connect(impl_->lightDialog, SIGNAL(attenuationChanged(virvo::vec3f const&)), impl_->canvas, SLOT(setLightAttenuation(virvo::vec3f const&)));

  connect(impl_->canvas, SIGNAL(newVolDesc(vvVolDesc*)), impl_->volInfoDialog, SLOT(onNewVolDesc(vvVolDesc*)));
  connect(impl_->canvas, SIGNAL(newVolDesc(vvVolDesc*)), impl_->sliceViewer, SLOT(onNewVolDesc(vvVolDesc*)));

  connect(impl_->timeStepDialog, SIGNAL(valueChanged(int)), impl_->canvas, SLOT(setTimeStep(int)));
  connect(impl_->timeStepDialog, SIGNAL(play(double)), impl_->canvas, SLOT(startAnimation(double)));
  connect(impl_->timeStepDialog, SIGNAL(pause()), impl_->canvas, SLOT(stopAnimation()));
  connect(impl_->timeStepDialog, SIGNAL(back()), impl_->canvas, SLOT(decTimeStep()));
  connect(impl_->timeStepDialog, SIGNAL(fwd()), impl_->canvas, SLOT(incTimeStep()));
  connect(impl_->timeStepDialog, SIGNAL(first()), impl_->canvas, SLOT(firstTimeStep()));
  connect(impl_->timeStepDialog, SIGNAL(last()), impl_->canvas, SLOT(lastTimeStep()));
  connect(impl_->canvas, SIGNAL(currentFrame(int)), impl_->timeStepDialog, SLOT(setCurrentFrame(int)));
  connect(impl_->canvas, SIGNAL(currentFrame(int)), impl_->sliceViewer, SLOT(onNewFrame(int)));

  // shortcuts

  QShortcut* sc; // reassign for each shortcut, objects are ref-counted by Qt, anyway

  // rendering quality
  sc = new QShortcut(tr("+"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(incQuality()));

  sc = new QShortcut(tr("="), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(incQuality()));

  sc = new QShortcut(tr("-"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(decQuality()));

  // rendering
  sc = new QShortcut(tr("o"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleOrientation()));

  sc = new QShortcut(tr("b"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleBoundaries()));

  sc = new QShortcut(tr("c"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(togglePalette()));

  sc = new QShortcut(tr("f"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleFrameRate()));

  sc = new QShortcut(tr("t"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleNumTextures()));

  sc = new QShortcut(tr("i"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleInterpolation()));

  sc = new QShortcut(tr("p"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleProjectionType()));

  sc = new QShortcut(tr("r"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), impl_->canvas, SLOT(resetCamera()));

  // animation
  sc = new QShortcut(tr("a"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), impl_->timeStepDialog, SLOT(togglePlayback()));

  sc = new QShortcut(tr("n"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), impl_->timeStepDialog, SLOT(stepFwd()));

  sc = new QShortcut(tr("Shift+n"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), impl_->timeStepDialog, SLOT(stepBack()));

  // misc.
  sc = new QShortcut(tr("q"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(close()));

  // cannot be done in dialog ctors because signals/slots need to be connected first
  impl_->lightDialog->applySettings();

  statusBar()->showMessage(tr("Welcome to DeskVOX!"));
}

vvMainWindow::~vvMainWindow()
{
  vvDebugMsg::msg(1, "vvMainWindow::~vvMainWindow()");
}

void vvMainWindow::lateInitialization()
{
  impl_->prefDialog = new vvPrefDialog(impl_->canvas, this);

  connect(impl_->prefDialog, SIGNAL(rendererChanged(const std::string&, const vvRendererFactory::Options&)),
    impl_->canvas, SLOT(setRenderer(const std::string&, const vvRendererFactory::Options&)));
  connect(impl_->prefDialog, SIGNAL(parameterChanged(vvParameters::ParameterType, const vvParam&)),
    impl_->canvas, SLOT(setParameter(vvParameters::ParameterType, const vvParam&)));
  connect(impl_->prefDialog, SIGNAL(parameterChanged(vvRenderer::ParameterType, const vvParam&)),
    impl_->canvas, SLOT(setParameter(vvRenderer::ParameterType, const vvParam&)));
  connect(impl_->canvas, SIGNAL(rendererChanged(vvRenderer*)),
    impl_->prefDialog, SLOT(handleNewRenderer(vvRenderer*)));

  // late-initialization is a noisance..
  impl_->prefDialog->handleNewRenderer(impl_->canvas->getRenderer());

  impl_->prefDialog->applySettings();
}

void vvMainWindow::loadVolumeFile(const QString& filename)
{
  QByteArray ba = filename.toLatin1();
  vvVolDesc* vd = new vvVolDesc(ba.data());
  vvFileIO fio;
  switch (fio.loadVolumeData(vd, vvFileIO::ALL_DATA))
  {
  case vvFileIO::OK:
  {
    vvDebugMsg::msg(2, "Loaded file: ", ba.data());
    if (vd->range(0)[0] == 0.0f && vd->range(0)[1] == 1.0f)
    {
      vd->findAndSetRange();
    }
    // use default TF if none stored
    if (vd->tf[0].isEmpty())
    {
      vd->tf[0].setDefaultAlpha(0, vd->range(0)[0], vd->range(0)[1]);
      vd->tf[0].setDefaultColors((vd->getChan() == 1) ? 0 : 2, vd->range(0)[0], vd->range(0)[1]);
    }
    impl_->canvas->setVolDesc(vd);
    impl_->dimensionDialog->setInitialDist(vd->getDist());

    impl_->ui->menuRecentVolumes->clear();
    std::vector<std::string> recents = getRecentFiles();
    for (std::vector<std::string>::const_iterator it = recents.begin();
         it != recents.end(); ++it)
    {
      QAction* action = new QAction((*it).c_str(), this);
      connect(action, SIGNAL(triggered()), this, SLOT(onRecentVolumeTriggered()));
      impl_->ui->menuRecentVolumes->insertAction(*impl_->ui->menuRecentVolumes->actions().begin(), action);
    } 
    addRecentFile(filename);
    break;
  }
  case vvFileIO::FILE_NOT_FOUND:
    vvDebugMsg::msg(2, "File not found: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error loading file"), tr("File not found: ") + filename, QMessageBox::Ok);
    break;
  default:
    vvDebugMsg::msg(2, "Cannot load file: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error loading file"), tr("Cannot load file: ") + filename, QMessageBox::Ok);
    break;
  }
}

void vvMainWindow::mergeFiles(const QString& firstFile, const int num, const int increment, vvVolDesc::MergeType mergeType)
{
  vvDebugMsg::msg(1, "vvMainWindow::mergeFiles()");

  QByteArray ba = firstFile.toLatin1();
  vvVolDesc* vd = new vvVolDesc(ba.data());
  vvFileIO fio;
  switch (fio.mergeFiles(vd, num, increment, mergeType))
  {
  case vvFileIO::OK:
    vvDebugMsg::msg(2, "Loaded slice sequence: ", vd->getFilename());
    if (vd->range(0)[0] == 0.0f && vd->range(0)[1] == 1.0f)
    {
      vd->findAndSetRange();
    }
    // use default TF if non stored
    if (vd->tf[0].isEmpty())
    {
      vd->tf[0].setDefaultAlpha(0, vd->range(0)[0], vd->range(0)[1]);
      vd->tf[0].setDefaultColors((vd->getChan() == 1) ? 0 : 2, vd->range(0)[0], vd->range(0)[1]);
    }
    impl_->canvas->setVolDesc(vd);
    break;
  case vvFileIO::FILE_NOT_FOUND:
    vvDebugMsg::msg(2, "File not found: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error merging file"), tr("File not found: ") + firstFile, QMessageBox::Ok);
    break;
  default:
    vvDebugMsg::msg(2, "Cannot merge file: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error merging file"), tr("Cannot merge file: ") + firstFile, QMessageBox::Ok);
    break;
  }
}

void vvMainWindow::toggleOrientation()
{
  impl_->ui->actionShowOrientation->trigger();
}

void vvMainWindow::toggleBoundaries()
{
  impl_->ui->actionShowBoundaries->trigger();
}

void vvMainWindow::togglePalette()
{
  impl_->ui->actionShowPalette->trigger();
}

void vvMainWindow::toggleFrameRate()
{
  impl_->ui->actionShowFrameRate->trigger();
}

void vvMainWindow::toggleNumTextures()
{
  impl_->ui->actionShowNumTextures->trigger();
}

void vvMainWindow::toggleInterpolation()
{
  impl_->prefDialog->toggleInterpolation();
}

void vvMainWindow::toggleProjectionType()
{
  vvObjView::ProjectionType type = static_cast<vvObjView::ProjectionType>(impl_->canvas->getParameter(vvParameters::VV_PROJECTIONTYPE).asInt());

  if (type == vvObjView::PERSPECTIVE)
  {
    type = vvObjView::ORTHO;
  }
  else
  {
    type = vvObjView::PERSPECTIVE;
  }
  impl_->canvas->setParameter(vvParameters::VV_PROJECTIONTYPE, static_cast<int>(type));
}

void vvMainWindow::incQuality()
{
  impl_->prefDialog->scaleStillQuality(1.05f);
}

void vvMainWindow::decQuality()
{
  impl_->prefDialog->scaleStillQuality(0.95f);
}

void vvMainWindow::onLoadVolumeTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onLoadVolumeTriggered()");

  QSettings settings;

  QString caption = tr("Load Volume File");
  QString dir = settings.value("canvas/voldir").value<QString>();
  QString filter = tr("All Volume Files (*.rvf *.xvf *.avf *.tif *.tiff *.hdr *.volb);;"
    "3D TIF Files (*.tif,*.tiff);;"
    "ASCII Volume Files (*.avf);;"
    "Extended Volume Files (*.xvf);;"
    "Raw Volume Files (*.rvf);;");
  if (virvo::fileio::hasFeature("nifti"))
    filter += "NifTI-1 Files (*.nii,*.nii.gz);;";
  filter += tr("All Files (*)");
  QString filename = QFileDialog::getOpenFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    loadVolumeFile(filename);

    QDir dir = QFileInfo(filename).absoluteDir();
    settings.setValue("canvas/voldir", dir.path());
  }
  else
  {
    QMessageBox::warning(this, tr("Error loading file"), tr("File name is empty"), QMessageBox::Ok);
  }
}

void vvMainWindow::onReloadVolumeTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onReloadVolumeTriggered()");

  vvVolDesc* vd = impl_->canvas->getVolDesc();
  if (vd != NULL)
  {
    loadVolumeFile(vd->getFilename());
  }
}

void vvMainWindow::onRecentVolumeTriggered()
{
  QAction* action = dynamic_cast<QAction*>(sender());
  if (action != NULL)
  {
    QString filename = action->text();
    if (!filename.isEmpty())
    {
      loadVolumeFile(filename);

      QDir dir = QFileInfo(filename).absoluteDir();
      QSettings settings;
      settings.setValue("canvas/voldir", dir.path());
    }
  }
}

void vvMainWindow::onSaveVolumeAsTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onSaveVolumeTriggered()");

  QString caption = tr("Save Volume");
  QString dir;
  QString filter = tr("All Volume Files (*.xvf *.rvf *.avf);;"
    "Extended Volume Files (*.xvf);;"
    "Raw Volume Files (*.rvf);;"
    "ASCII Volume Files (*.avf);;"
    "All Files (*)");
  QString filename = QFileDialog::getSaveFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    vvFileIO fio;
    QByteArray ba = filename.toLatin1();
    impl_->canvas->getVolDesc()->setFilename(ba.data());
    switch (fio.saveVolumeData(impl_->canvas->getVolDesc(), true))
    {
    case vvFileIO::OK:
      vvDebugMsg::msg(2, "Volume saved as ", impl_->canvas->getVolDesc()->getFilename());
      break;
    default:
      vvDebugMsg::msg(0, "Unhandled error saving ", impl_->canvas->getVolDesc()->getFilename());
      break;
    }
  }
}

void vvMainWindow::onMergeFilesTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onMergeFilesTriggered()");

  if (impl_->mergeDialog->exec() == QDialog::Accepted)
  {
    const QString filename = impl_->mergeDialog->getFilename();

    int numFiles = 0;
    if (impl_->mergeDialog->numFilesLimited())
    {
      numFiles = impl_->mergeDialog->getNumFiles();
    }

    int increment = 0;
    if (impl_->mergeDialog->filesNumbered())
    {
      increment = impl_->mergeDialog->getFileIncrement();
    }

    const vvVolDesc::MergeType mergeType = impl_->mergeDialog->getMergeType();

    mergeFiles(filename, numFiles, increment, mergeType);
  }
}

void vvMainWindow::onLoadCameraTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onLoadCameraTriggered()");

  QString caption = tr("Load Camera File");
  QString dir;
  QString filter = tr("Camera Files (*cam);;"
    "All Files (*)");
  QString filename = QFileDialog::getOpenFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    impl_->canvas->loadCamera(filename);
  }
}

void vvMainWindow::onSaveCameraAsTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onSaveCameraAsTriggered()");

  QString caption = tr("Save Camera to File");
  QString dir = "camera.cam";
  QString filter = tr("Camera Files (*cam);;"
    "All Files (*)");
  QString filename = QFileDialog::getSaveFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    impl_->canvas->saveCamera(filename);
  }
}

void vvMainWindow::onScreenshotTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onScreenshotTriggered()");

  impl_->screenshotDialog->raise();
  impl_->screenshotDialog->show();
}

void vvMainWindow::onPreferencesTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onPreferencesTriggered()");

  impl_->prefDialog->raise();
  impl_->prefDialog->show();
}

void vvMainWindow::onTransferFunctionTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onTransferFunctionTriggered()");

  impl_->tfDialog->raise();
  impl_->tfDialog->show();
}

void vvMainWindow::onClippingPlaneTriggered()
{
  impl_->clipDialog->raise();
  impl_->clipDialog->show();
}

void vvMainWindow::onLightSourceTriggered()
{
  impl_->lightDialog->raise();
  impl_->lightDialog->show();
}

void vvMainWindow::onBackgroundColorTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onBackgroundColorTriggered()");

  vvColor bgcolor = impl_->canvas->getParameter(vvParameters::VV_BG_COLOR);
  QColor initial;
  initial.setRedF(bgcolor[0]);
  initial.setGreenF(bgcolor[1]);
  initial.setBlueF(bgcolor[2]);
  QColor qcolor = QColorDialog::getColor(initial);
  if (qcolor.isValid())
  {
    vvColor color(qcolor.redF(), qcolor.greenF(), qcolor.blueF());
    impl_->canvas->setParameter(vvParameters::VV_BG_COLOR, color);
    QSettings settings;
    settings.setValue("canvas/bgcolor", qcolor);
  }
}

void vvMainWindow::onSampleDistancesTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onSampleDistancesTriggered()");

  impl_->dimensionDialog->raise();
  impl_->dimensionDialog->show();
}

void vvMainWindow::onShowOrientationTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowOrientationTriggered()");

  impl_->canvas->setParameter(vvRenderState::VV_ORIENTATION, checked);
}

void vvMainWindow::onShowBoundariesTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowBoundariesTriggered()");

  impl_->canvas->setParameter(vvRenderState::VV_BOUNDARIES, checked);
}

void vvMainWindow::onShowPaletteTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowPaletteTriggered()");

  impl_->canvas->setParameter(vvRenderState::VV_PALETTE, checked);
}

void vvMainWindow::onShowNumTexturesTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowNumTexturesTriggered()");

  impl_->canvas->setParameter(vvRenderState::VV_QUALITY_DISPLAY, checked);
}

void vvMainWindow::onShowFrameRateTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowFrameRateTriggered()");

  impl_->canvas->setParameter(vvRenderState::VV_FPS_DISPLAY, checked);
}

void vvMainWindow::onAutoRotationTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onAutoRotationTriggered()");

  impl_->canvas->setParameter(vvParameters::VV_SPIN_ANIMATION, checked);
}

void vvMainWindow::onVolumeInformationTriggered()
{
  impl_->volInfoDialog->raise();
  impl_->volInfoDialog->show();
}

void vvMainWindow::onSliceViewerTriggered()
{
  impl_->sliceViewer->raise();
  impl_->sliceViewer->show();
}

void vvMainWindow::onTimeStepsTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onTimeStepsTriggered()");

  impl_->timeStepDialog->raise();
  impl_->timeStepDialog->show();
}

void vvMainWindow::onKeyboardCommandsClicked()
{
  vvDebugMsg::msg(3, "vvMainWindow::onKeyboardCommandsClicked()");

  impl_->shortcutDialog->raise();
  impl_->shortcutDialog->show();
}

void vvMainWindow::onNewVolDesc(vvVolDesc* vd)
{
  vvDebugMsg::msg(3, "vvMainWindow::onNewVolDesc()");

  impl_->timeStepDialog->setFrames(vd ? vd->frames : 0);
}

void vvMainWindow::onStatusMessage(const std::string& str)
{
  vvDebugMsg::msg(3, "vvMainWindow::onStatusMessage()");

  statusBar()->showMessage(str.c_str());
}

int main(int argc, char** argv)
{
  vvDebugMsg::setDebugLevel(0);

  QApplication a(argc, argv);

  QCoreApplication::setOrganizationName("DeskVOX");
  QCoreApplication::setApplicationName("DeskVOX");

  // parse command line
  QString filename;
  QSize size(600, 600);
  QStringList arglist = a.arguments();
  for (QStringList::iterator it = arglist.begin();
       it != arglist.end(); ++it)
  {
    QString str = *it;
    if (str == arglist.first())
    {
      continue;
    }

    if (str == "-size")
    {
      ++it;
      int w = 0;
      int h = 0;;
      if (it != arglist.end())
      {
        str = *it;
        w = str.toInt();
      }
      else
      {
        vvDebugMsg::msg(0, "Warning: -size followed by no arguments");
        break;
      }

      ++it;
      if (it != arglist.end())
      {
        str = *it;
        h = str.toInt();
        size = QSize(w, h);
      }
      else
      {
        vvDebugMsg::msg(0, "Warning: -size followed by only one argument");
        break;
      }
    }
    else if (str[0] == '-')
    {
      vvDebugMsg::msg(0, "Warning: invalid command line option");
      break;
    }
    else
    {
      filename = str;
    }
  }

  // create main window
  vvMainWindow win(filename);
  win.resize(size);
  win.show();
  win.lateInitialization();
  return a.exec();
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
