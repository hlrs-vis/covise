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
#include "vvtfdialog.h"

#include "tfeditor/colorbox.h"
#include "tfeditor/gaussianbox.h"
#include "tfeditor/graphicsscene.h"
#include "tfeditor/pyramidbox.h"
#include "tfeditor/skipbox.h"

#include "ui_vvtfdialog.h"

#include <virvo/math/math.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvvoldesc.h>

#include <QFileDialog>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>

#include <boost/bimap.hpp>

#include <algorithm>
#include <vector>

using namespace virvo;


namespace
{

static const float PIN_WIDTH = 2.0f;
static const float SELECTED_WIDTH = 4.0f;
static const size_t COLORBAR_HEIGHT = 30;
static const size_t TF_WIDTH = 768;
static const size_t TF_HEIGHT = 256;
static const size_t INVAL_PIN = size_t(-1);

/** Convert canvas x coordinates to data values.
  @param canvas canvas x coordinate [0..1]
  @return data value
*/
float norm2data(vec2f const& zoomrange, float canvas)
{
  return canvas * (zoomrange[1] - zoomrange[0]) + zoomrange[0];
}

/** Convert data value to x coordinate in TF canvas.
  @param data data value
  @return canvas x coordinate [0..1]
*/
float data2norm(vec2f const& zoomrange, float data)
{
  return (data - zoomrange[0]) / (zoomrange[1] - zoomrange[0]);
}

/** Convert horizontal differences on the canvas to data differences.
*/
float normd2datad(vec2f const& zoomrange, float canvas)
{
  return canvas * (zoomrange[1] - zoomrange[0]);
}

#if 0 // not used
/** Convert differences in data to the canvas.
*/
float datad2normd(vec2f const& zoomrange, float data)
{
  return data / (zoomrange[1] - zoomrange[0]);
}
#endif
}

typedef QGraphicsRectItem Pin;

struct vvTFDialog::Impl
{
  Impl()
    : ui(new Ui::TFDialog)
    , colorscene(new MouseGraphicsScene)
    , coloritem(new QGraphicsPixmapItem)
    , alphascene(new MouseGraphicsScene)
    , alphaitem(new QGraphicsPixmapItem)
    , vd(NULL)
    , colorPinSelected(INVAL_PIN)
    , alphaPinSelected(INVAL_PIN)
    , colorPinMoving(INVAL_PIN)
    , alphaPinMoving(INVAL_PIN)
    , zoomRange(vec2f(0.0f, 1.0f))
    , logHistogram(true)
    , colorDirty(true)
    , alphaDirty(true)
    , histDirty(true)
  {
    colorscene->addItem(coloritem);
    alphascene->addItem(alphaitem);
  }

  ~Impl()
  {
    delete coloritem;
    delete alphaitem;
    delete colorscene;
    delete alphascene;
  }

  Pin* getSelectedPin() const
  {
    if (colorPinSelected != INVAL_PIN && colorPinSelected < colorpins.size())
      return colorpins[colorPinSelected];

    if (alphaPinSelected != INVAL_PIN && alphaPinSelected < alphapins.size())
      return alphapins[alphaPinSelected];

    return NULL;
  }

  Pin* getMovingPin() const
  {
    if (colorPinMoving != INVAL_PIN && colorPinMoving < colorpins.size())
      return colorpins[colorPinMoving];

    if (alphaPinMoving != INVAL_PIN && alphaPinMoving < alphapins.size())
      return alphapins[alphaPinMoving];

    return NULL;
  }

  std::auto_ptr<Ui::TFDialog> ui;

  MouseGraphicsScene* colorscene;
  QGraphicsPixmapItem* coloritem;
  std::vector<Pin*> colorpins;

  MouseGraphicsScene* alphascene;
  QGraphicsPixmapItem* alphaitem;
  std::vector<Pin*> alphapins;

  vvVolDesc* vd;

  size_t colorPinSelected;
  size_t alphaPinSelected;
  size_t colorPinMoving;
  size_t alphaPinMoving;

  typedef boost::bimap< Pin*, vvTFWidget*> bm_type;
  bm_type pin2widget;

  vec2f zoomRange; ///< min/max for zoom area on data range

  bool logHistogram;

  // texture data
  std::vector<uchar> colorTex;
  std::vector<uchar> alphaTex;
  std::vector<uchar> histTex;

  bool colorDirty;
  bool alphaDirty;
  bool histDirty;

private:

  Impl(Impl const& rhs);
  Impl& operator=(Impl const& rhs);

};

vvTFDialog::vvTFDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
  , _canvas(canvas)
{
  vvDebugMsg::msg(1, "vvTFDialog::vvTFDialog()");

  impl_->ui->setupUi(this);

  impl_->ui->color1DView->setScene(impl_->colorscene);
  impl_->ui->alpha1DView->setScene(impl_->alphascene);

  connect(impl_->ui->colorButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->pyramidButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->gaussianButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->customButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->skipRangeButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->deleteButton, SIGNAL(clicked()), this, SLOT(onDeleteClicked()));
  connect(impl_->ui->undoButton, SIGNAL(clicked()), this, SLOT(onUndoClicked()));
  connect(impl_->ui->logHistBox, SIGNAL(toggled(bool)), this, SLOT(onLogHistToggled(bool)));
  connect(impl_->ui->zoomMinBox, SIGNAL(valueChanged(double)), this, SLOT(onZoomMinChanged(double)));
  connect(impl_->ui->zoomMaxBox, SIGNAL(valueChanged(double)), this, SLOT(onZoomMaxChanged(double)));
  connect(impl_->ui->presetColorsBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetColorsChanged(int)));
  connect(impl_->ui->presetAlphaBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetAlphaChanged(int)));
  connect(impl_->ui->discrSlider, SIGNAL(valueChanged(int)), this, SLOT(onDiscrChanged(int)));
  connect(impl_->ui->applyButton, SIGNAL(clicked()), this, SLOT(onApplyClicked()));
  connect(_canvas, SIGNAL(newVolDesc(vvVolDesc*)), this, SLOT(onNewVolDesc(vvVolDesc*)));
  connect(impl_->ui->saveButton, SIGNAL(clicked()), this, SLOT(saveTF()));
  connect(impl_->ui->loadButton, SIGNAL(clicked()), this, SLOT(loadTF()));
  connect(impl_->colorscene, SIGNAL(mouseMoved(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseMove(QPointF, Qt::MouseButton)));
  connect(impl_->colorscene, SIGNAL(mousePressed(QPointF, Qt::MouseButton)), this, SLOT(onTFMousePress(QPointF, Qt::MouseButton)));
  connect(impl_->colorscene, SIGNAL(mouseReleased(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseRelease(QPointF, Qt::MouseButton)));
  connect(impl_->alphascene, SIGNAL(mouseMoved(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseMove(QPointF, Qt::MouseButton)));
  connect(impl_->alphascene, SIGNAL(mousePressed(QPointF, Qt::MouseButton)), this, SLOT(onTFMousePress(QPointF, Qt::MouseButton)));
  connect(impl_->alphascene, SIGNAL(mouseReleased(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseRelease(QPointF, Qt::MouseButton)));
}

vvTFDialog::~vvTFDialog()
{
  vvDebugMsg::msg(1, "vvTFDialog::~vvTFDialog()");
}

void vvTFDialog::drawTF()
{
  // color bar
  if (impl_->colorDirty)
  {
    int w = impl_->ui->color1DView->width();
    int h = impl_->ui->color1DView->height();
    impl_->colorTex.resize(w * h * 4);
    makeColorBar(&impl_->colorTex, w);
  }

  // histogram
  if (impl_->histDirty)
  {
    int w = impl_->ui->alpha1DView->width();
    int h = impl_->ui->alpha1DView->height();
    impl_->histTex.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    makeHistogramTexture(&impl_->histTex, w, h);
  }

  // alpha widgets
  if (impl_->alphaDirty || impl_->histDirty /*TODO!*/)
  {
    int w = impl_->ui->alpha1DView->width();
    int h = impl_->ui->alpha1DView->height();
    impl_->alphaTex.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    makeAlphaTexture(&impl_->alphaTex, w, h);
  }

  // blend histogram texture and alpha texture
  if (impl_->histDirty || impl_->alphaDirty)
  {
    // TODO: use hardware for blending
    int w = impl_->ui->alpha1DView->width();
    int h = impl_->ui->alpha1DView->height();

    std::vector<uchar> dest(impl_->alphaTex.size());

    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        const uchar* histptr = impl_->histTex.data() + ((h - y - 1) * w + x) * 4; // TODO: why only flip hist and not alpha texture?
        const uchar* alphaptr = impl_->alphaTex.data() + (y * w + x) * 4;
        uchar* destptr = dest.data() + (y * w + x) * 4;

        destptr[0] = (uchar) ((float)alphaptr[0] * (float)alphaptr[3] + (float)histptr[0] * (1.f - (float)alphaptr[3]));
        destptr[1] = (uchar) ((float)alphaptr[1] * (float)alphaptr[3] + (float)histptr[1] * (1.f - (float)alphaptr[3]));
        destptr[2] = (uchar) ((float)alphaptr[2] * (float)alphaptr[3] + (float)histptr[2] * (1.f - (float)alphaptr[3]));
        destptr[3] = (uchar) ((float)alphaptr[3] * (float)alphaptr[3] + (float)histptr[3] * (1.f - (float)alphaptr[3]));
      }
    }
    std::copy(dest.begin(), dest.end(), impl_->alphaTex.begin());
  }

  drawColorTexture();
  drawAlphaTexture();

  impl_->colorDirty = false;
  impl_->histDirty = false;
  impl_->alphaDirty = false;

  impl_->colorscene->invalidate();
  impl_->alphascene->invalidate();
}

void vvTFDialog::drawColorTexture()
{
  vvDebugMsg::msg(3, "vvTFDialog::drawColorTexture()");

  int w = impl_->ui->color1DView->width();
  int h = impl_->ui->color1DView->height();

  QImage img(&impl_->colorTex[0], w, h, QImage::Format_ARGB32);
  img = img.scaled(QSize(w, h * COLORBAR_HEIGHT / 3));
  if (!img.isNull())
  {
    QPixmap colorpm = QPixmap::fromImage(img);
    impl_->coloritem->setPixmap(colorpm);
  }
}

void vvTFDialog::drawAlphaTexture()
{
  vvDebugMsg::msg(3, "vvTFDialog::drawAlphaTexture()");

  int w = impl_->ui->alpha1DView->width();
  int h = impl_->ui->alpha1DView->height();

  QImage img(&impl_->alphaTex[0], w, h, QImage::Format_ARGB32);

  if (!img.isNull())
  {
    QPixmap alphapm = QPixmap::fromImage(img);
    impl_->alphaitem->setPixmap(alphapm);
  }
}

void vvTFDialog::clearPins()
{
  for (std::vector<Pin*>::const_iterator it = impl_->colorpins.begin();
       it != impl_->colorpins.end(); ++it)
  {
    impl_->colorscene->removeItem(*it);
    delete *it;
  }
  impl_->colorpins.clear();

  for (std::vector<Pin*>::const_iterator it = impl_->alphapins.begin();
       it != impl_->alphapins.end(); ++it)
  {
    impl_->alphascene->removeItem(*it);
    delete *it;
  }
  impl_->alphapins.clear();

  impl_->pin2widget.clear();
}

void vvTFDialog::createPins()
{
  if (!_canvas->getVolDesc())
    return;

  for (std::vector<vvTFWidget*>::const_iterator it = _canvas->getVolDesc()->tf[0]._widgets.begin();
       it != _canvas->getVolDesc()->tf[0]._widgets.end(); ++it)
  {
    createPin(*it);
  }
}

void vvTFDialog::createPin(vvTFWidget* w)
{
  bool selected = false; // TODO
//  bool mouseover = false;
  float rectw = selected ? SELECTED_WIDTH : PIN_WIDTH;
  float xpos = data2norm(impl_->zoomRange, w->_pos[0]) * static_cast<float>(TF_WIDTH);

  Pin* pin = NULL;
  if (dynamic_cast<vvTFColor*>(w) != NULL) // draw color pin
  {
    pin = impl_->colorscene->addRect(-rectw * 0.5f, 0, rectw, COLORBAR_HEIGHT - 1, QPen(), QBrush(Qt::SolidPattern));
    pin->setPos(xpos, 0);
    impl_->colorpins.push_back(pin);
  }
  else if ((dynamic_cast<vvTFPyramid*>(w) != NULL) ||
           (dynamic_cast<vvTFBell*>(w) != NULL) ||
           (dynamic_cast<vvTFSkip*>(w) != NULL) ||
           (dynamic_cast<vvTFCustom*>(w) != NULL)) // draw alpha pin
  {
    pin = impl_->alphascene->addRect(-rectw * 0.5f, 0, rectw, TF_HEIGHT - COLORBAR_HEIGHT, QPen(), QBrush(Qt::SolidPattern));
    pin->setPos(xpos, 0);
    impl_->alphapins.push_back(pin);

  }
  impl_->pin2widget.insert(Impl::bm_type::value_type(pin, w));
}

void vvTFDialog::makeColorBar(std::vector<uchar>* colorBar, int width) const
{
  if (!_canvas->getVolDesc())
  {
    memset(&(*colorBar)[0], 0, colorBar->size());
    return;
  }

  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    // BGRA to fit QImage's little endian ARGB32 format
    _canvas->getVolDesc()->tf[0].makeColorBar(width, &(*colorBar)[0], impl_->zoomRange[0], impl_->zoomRange[1], false, vvToolshed::VV_BGRA);
  }
}

void vvTFDialog::makeHistogramTexture(std::vector<uchar>* hist, int width, int height) const
{
  assert(_canvas->getVolDesc());

  size_t size[] = { static_cast<size_t>(width), static_cast<size_t>(height) };
  vvColor col(0.4f, 0.4f, 0.4f);

  _canvas->getVolDesc()->makeHistogramTexture(0, 0, 1, size, hist->data(),
                                              impl_->logHistogram ? vvVolDesc::VV_LOGARITHMIC
                                                                  : vvVolDesc::VV_LINEAR,
                                              &col, impl_->zoomRange[0], impl_->zoomRange[1]);
}

void vvTFDialog::makeAlphaTexture(std::vector<uchar>* alphaTex, int width, int height) const
{
  if (!_canvas->getVolDesc())
  {
    memset(&(*alphaTex)[0], 0, alphaTex->size());
    return;
  }

  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    // BGRA to fit QImage's little endian ARGB32 format
    _canvas->getVolDesc()->tf[0].makeAlphaTexture(width, height, &(*alphaTex)[0], impl_->zoomRange[0], impl_->zoomRange[1], vvToolshed::VV_BGRA);
  }
}

void vvTFDialog::onUndoClicked()
{
  impl_->colorDirty = true;
  impl_->alphaDirty = true;
  emit undo();
  emit newTransferFunction();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onLogHistToggled(bool value)
{
  impl_->logHistogram = value;
  impl_->histDirty = true;
  drawTF();
}

void vvTFDialog::onNewWidget()
{
  vvTFWidget* widget = NULL;

  if (QObject::sender() == impl_->ui->colorButton)
  {
    widget = new vvTFColor(vvColor(), norm2data(impl_->zoomRange, 0.5f));
    impl_->colorDirty = true;
  }
  else if (QObject::sender() == impl_->ui->pyramidButton)
  {
    widget = new vvTFPyramid(vvColor(), false, 1.0f, norm2data(impl_->zoomRange, 0.5f), normd2datad(impl_->zoomRange, 0.4f), normd2datad(impl_->zoomRange, 0.2f));
    impl_->alphaDirty = true;
  }
  else if (QObject::sender() == impl_->ui->gaussianButton)
  {
    widget = new vvTFBell(vvColor(), false, 1.0f, norm2data(impl_->zoomRange, 0.5f), normd2datad(impl_->zoomRange, 0.2f));
    impl_->alphaDirty = true;
  }
  else if (QObject::sender() == impl_->ui->customButton)
  {
    widget = new vvTFCustom(norm2data(impl_->zoomRange, 0.5f), norm2data(impl_->zoomRange, 0.5f));
    impl_->alphaDirty = true;
  }
  else if (QObject::sender() == impl_->ui->skipRangeButton)
  {
    widget = new vvTFSkip(norm2data(impl_->zoomRange, 0.5f), normd2datad(impl_->zoomRange, 0.2f));
    impl_->alphaDirty = true;
  }

  emit newWidget(widget);
  createPin(widget);
  updateSettingsBox();
  drawTF();
}

void vvTFDialog::onDeleteClicked()
{
  if (_canvas->getVolDesc()->tf[0]._widgets.size() == 0 || impl_->getSelectedPin() == NULL)
  {
    return;
  }
  _canvas->getVolDesc()->tf[0].putUndoBuffer();

  Pin* selected = impl_->getSelectedPin();

  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  _canvas->getVolDesc()->tf[0]._widgets.erase
  (
    std::find(_canvas->getVolDesc()->tf[0]._widgets.begin(), _canvas->getVolDesc()->tf[0]._widgets.end(), lit->second)
  );
  impl_->colorPinSelected = INVAL_PIN;
  impl_->alphaPinSelected = INVAL_PIN;
  impl_->colorDirty = true;
  impl_->alphaDirty = true;

  emitTransFunc();
  clearPins();
  createPins();
  updateSettingsBox();
  drawTF();
}

void vvTFDialog::onPresetColorsChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetColorsChanged()");

  impl_->colorDirty = true;
  _canvas->getVolDesc()->tf[0].setDefaultColors(index, impl_->zoomRange[0], impl_->zoomRange[1]);
  emitTransFunc();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onPresetAlphaChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetAlphaChanged()");

  impl_->alphaDirty = true;
  _canvas->getVolDesc()->tf[0].setDefaultAlpha(index, impl_->zoomRange[0], impl_->zoomRange[1]);
  emitTransFunc();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onDiscrChanged(int num)
{
  impl_->alphaDirty = true;
  impl_->ui->discrLabel->setText(QString::number(num));
  _canvas->getVolDesc()->tf[0].setDiscreteColors(num);
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onApplyClicked()
{
  drawTF();
}

void vvTFDialog::onNewVolDesc(vvVolDesc *vd)
{
  impl_->vd = vd;
  impl_->colorDirty = true;
  impl_->histDirty = true;
  impl_->alphaDirty = true;
  clearPins();
  createPins();
  if (vd != NULL)
  {
    impl_->ui->minLabel->setText(QString::number(vd->range(0).x));
    impl_->ui->maxLabel->setText(QString::number(vd->range(0).y));

    impl_->ui->zoomMinBox->setMinimum(vd->mapping(0).x);
    impl_->ui->zoomMinBox->setMaximum(vd->mapping(0).y);
    impl_->ui->zoomMinBox->setValue(vd->range(0).x);

    impl_->ui->zoomMaxBox->setMinimum(vd->mapping(0).x);
    impl_->ui->zoomMaxBox->setMaximum(vd->mapping(0).y);
    impl_->ui->zoomMaxBox->setValue(vd->range(0).y);

    impl_->ui->discrSlider->setValue(vd->tf[0].getDiscreteColors());
  }
  drawTF();
}

void vvTFDialog::onZoomMinChanged(double zm)
{
  assert(impl_->vd != NULL);

  // Convert to [0..1]
  // TODO...
  impl_->zoomRange[0] = zm;
  impl_->colorDirty = true;
  impl_->histDirty = true;
  impl_->alphaDirty = true;
  clearPins();
  createPins();

  drawTF();
}

void vvTFDialog::onZoomMaxChanged(double zm)
{
  assert(impl_->vd != NULL);

  // Convert to [0..1]
  // TODO...
  impl_->zoomRange[1] = zm;
  impl_->colorDirty = true;
  impl_->histDirty = true;
  impl_->alphaDirty = true;
  clearPins();
  createPins();

  drawTF();
}

void vvTFDialog::saveTF()
{
  QString caption = tr("Save Transfer Function");
  QString dir;
  QString filter = tr("Transfer function files (*.vtf);;"
    "All Files (*)");
  QString filename = QFileDialog::getSaveFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    std::string strfn = filename.toStdString();
    _canvas->getVolDesc()->tf[0].save(strfn.c_str());
  }
}

void vvTFDialog::loadTF()
{
  QString caption = tr("Load Transfer Function");
  QString dir;
  QString filter = tr("Transfer function files (*.vtf);;"
    "All Files (*)");
  QString filename = QFileDialog::getOpenFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    std::string strfn = filename.toStdString();
    _canvas->getVolDesc()->tf[0].load(strfn.c_str());
  }
  emitTransFunc();
  clearPins();
  createPins();
  impl_->ui->discrSlider->setValue(_canvas->getVolDesc()->tf[0].getDiscreteColors());
  drawTF();
}

void vvTFDialog::onColor(const QColor& color)
{
  Pin* selected = impl_->getSelectedPin();
  assert(selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* wid = lit->second;
  if (vvTFColor* c = dynamic_cast<vvTFColor*>(wid))
  {
    c->setColor(vvColor(color.redF(), color.greenF(), color.blueF()));
  }
  else if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setColor(vvColor(color.redF(), color.greenF(), color.blueF()));
  }
  else if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid))
  {
    p->setColor(vvColor(color.redF(), color.greenF(), color.blueF()));
  }
  else
  {
    assert(false);
  }
  impl_->colorDirty = true;
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onHasOwnColor(bool hascolor)
{
  Pin* selected = impl_->getSelectedPin();
  assert(selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* wid = lit->second;
  if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid))
  {
    p->setOwnColor(hascolor);
  }
  else if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setOwnColor(hascolor);
  }
  else
  {
    assert(false);
  }
  impl_->colorDirty = true;
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onOpacity(float opacity)
{
  Pin* selected = impl_->getSelectedPin();
  assert(selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* wid = lit->second;
  if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setOpacity(opacity);
  }
  else if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid))
  {
    p->setOpacity(opacity);
  }
  else
  {
    assert(false);
  }
  impl_->colorDirty = true;
  impl_->alphaDirty = true;
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onSize(vec3f const& size)
{
  Pin* selected = impl_->getSelectedPin();
  assert(selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* wid = lit->second;
  if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setSize(size);
  }
  else if (vvTFSkip* s = dynamic_cast<vvTFSkip*>(wid))
  {
    s->setSize(size);
  }
  else
  {
    assert(false);
  }
  impl_->colorDirty = true;
  impl_->alphaDirty = true;
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onTop(vec3f const& top)
{
  Pin* selected = impl_->getSelectedPin();
  assert(selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* wid = lit->second;
  vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid);
  assert(p != NULL);
  p->setTop(top);
  impl_->colorDirty = true;
  impl_->alphaDirty = true;
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onBottom(vec3f const& bottom)
{
  Pin* selected = impl_->getSelectedPin();
  assert(selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* wid = lit->second;
  vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid);
  assert(p != NULL);
  p->setBottom(bottom);
  impl_->colorDirty = true;
  impl_->alphaDirty = true;
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onTFMouseMove(QPointF pos, Qt::MouseButton /* button */)
{
  std::vector<Pin*>& pins = QObject::sender() == impl_->colorscene ? impl_->colorpins : impl_->alphapins;

  float posX01 = pos.x() / (impl_->ui->color1DView->width() - 1);
  float posXZoom = lerp(impl_->zoomRange.x, impl_->zoomRange.y, posX01);
  // Value under cursor
  impl_->ui->valueLabel->setText("Value: " + QString::number(posXZoom));

  if (impl_->getMovingPin() != NULL)
  {
    Pin* pin = impl_->getMovingPin();
    if (pin == NULL)
    {
      return;
    }

    float x = static_cast<float>(pos.x());
    x = ts_clamp(x, 0.0f, static_cast<float>(TF_WIDTH));
    QPointF pinpos = pin->pos();
    pin->setPos(x, pinpos.y());
    Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(pin);
    vvTFWidget* w = lit->second;
    if (w != NULL)
    {
      x /= static_cast<float>(TF_WIDTH);
      x = norm2data(impl_->zoomRange, x);
      vec3f oldpos = w->pos();
      w->setPos(x, oldpos[1], oldpos[2]);
    }
    impl_->colorDirty = true;
    impl_->alphaDirty = true;
    emitTransFunc();
    drawTF();
  }
  else
  {
    for (std::vector<Pin*>::const_iterator it = pins.begin();
         it != pins.end(); ++it)
    {
      float minx = (*it)->pos().x() - PIN_WIDTH * 0.5f - 1;
      float maxx = (*it)->pos().x() + PIN_WIDTH * 0.5f + 1;
      if (pos.x() >= minx && pos.x() <= maxx)
      {
        // TODO: add highlight
      }
      else
      {
        // TODO: remove highlight
      }
    }
  }
}

void vvTFDialog::onTFMousePress(QPointF pos, Qt::MouseButton button)
{
  const std::vector<Pin*>& pins = QObject::sender() == impl_->colorscene ? impl_->colorpins : impl_->alphapins;

  if (button == Qt::LeftButton)
  {
    if (QObject::sender() == impl_->colorscene)
      impl_->colorPinSelected = INVAL_PIN;
    else if (QObject::sender() == impl_->alphascene)
      impl_->alphaPinSelected = INVAL_PIN;

    size_t idx;
    std::vector<Pin*>::const_iterator it;
    for (it = pins.begin(), idx = 0; it != pins.end(); ++it, ++idx)
    {
      float minx = (*it)->pos().x() - PIN_WIDTH * 0.5f - 1;
      float maxx = (*it)->pos().x() + PIN_WIDTH * 0.5f + 1;
      if (pos.x() >= minx && pos.x() <= maxx)
      {
        if (QObject::sender() == impl_->colorscene)
        {
          impl_->colorPinSelected = idx;
          impl_->colorPinMoving = idx;
        }
        else if (QObject::sender() == impl_->alphascene)
        {
          impl_->alphaPinSelected = idx;
          impl_->alphaPinMoving = idx;
        }
        break;
      }
    }
    updateSettingsBox();
  }
}

void vvTFDialog::onTFMouseRelease(QPointF /* pos */, Qt::MouseButton /* button */)
{
  impl_->colorPinMoving = INVAL_PIN;
  impl_->alphaPinMoving = INVAL_PIN;
}

void vvTFDialog::emitTransFunc()
{
  emit newTransferFunction();
}

void vvTFDialog::updateSettingsBox()
{
  // clear settings layout
  if (QLayoutItem* item = impl_->ui->settingsLayout->takeAt(0))
  {
    QWidget* widget = item->widget();
    delete widget;
  }

  Pin* selected = impl_->getSelectedPin();
  if (selected == NULL)
  {
    return;
  }

  // new settings box
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* w = lit->second;
  if (vvTFColor* c = dynamic_cast<vvTFColor*>(w))
  {
    tf::ColorBox* cb = new tf::ColorBox(this);
    impl_->ui->settingsLayout->addWidget(cb);
    cb->setColor(c->color());
    connect(cb, SIGNAL(color(const QColor&)), this, SLOT(onColor(const QColor&)));
  }
  else if (vvTFBell* b = dynamic_cast<vvTFBell*>(w))
  {
    tf::GaussianBox* gb = new tf::GaussianBox(this);
    impl_->ui->settingsLayout->addWidget(gb);
    gb->setZoomRange(impl_->zoomRange);
    gb->setHasColor(b->hasOwnColor());
    gb->setColor(b->color());
    gb->setSize(b->size());
    gb->setOpacity(b->opacity());
    connect(gb, SIGNAL(color(const QColor&)), this, SLOT(onColor(const QColor&)));
    connect(gb, SIGNAL(hasColor(bool)), this, SLOT(onHasOwnColor(bool)));
    connect(gb, SIGNAL(size(virvo::vec3f const&)), this, SLOT(onSize(virvo::vec3f const&)));
    connect(gb, SIGNAL(opacity(float)), this, SLOT(onOpacity(float)));
  }
  else if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(w))
  {
    tf::PyramidBox* pb = new tf::PyramidBox(this);
    impl_->ui->settingsLayout->addWidget(pb);
    pb->setZoomRange(impl_->zoomRange);
    pb->setHasColor(p->hasOwnColor());
    pb->setColor(p->color());
    pb->setTop(p->top());
    pb->setBottom(p->bottom());
    pb->setOpacity(p->opacity());
    connect(pb, SIGNAL(color(const QColor&)), this, SLOT(onColor(const QColor&)));
    connect(pb, SIGNAL(hasColor(bool)), this, SLOT(onHasOwnColor(bool)));
    connect(pb, SIGNAL(top(virvo::vec3f const&)), this, SLOT(onTop(virvo::vec3f const&)));
    connect(pb, SIGNAL(bottom(virvo::vec3f const&)), this, SLOT(onBottom(virvo::vec3f const&)));
    connect(pb, SIGNAL(opacity(float)), this, SLOT(onOpacity(float)));
  }
  else if (vvTFSkip* s = dynamic_cast<vvTFSkip*>(w))
  {
    tf::SkipBox* sb = new tf::SkipBox(this);
    impl_->ui->settingsLayout->addWidget(sb);
    sb->setZoomRange(impl_->zoomRange);
    sb->setSize(s->size());
    connect(sb, SIGNAL(size(virvo::vec3f const&)), this, SLOT(onSize(virvo::vec3f const&)));
  }
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
