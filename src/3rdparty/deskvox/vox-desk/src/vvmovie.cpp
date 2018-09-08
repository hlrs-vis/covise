// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
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

#include <string.h>

#include <fstream>

// Virvo:
#include <virvo/math/math.h>
#include <vvdebugmsg.h>
#include <vvtokenizer.h>
#include <vvtoolshed.h>
#include <vvrenderer.h>
#include <vvtoolshed.h>
#include <vvfileio.h>

// Local:
#include "vvcanvas.h"
#include "vvmovie.h"

using namespace virvo;
using namespace vox;
using namespace std;

//============================================================================
// Class Definitions
//============================================================================

//----------------------------------------------------------------------------
/// Constructor
vvMovie::vvMovie(vvCanvas* canvas)
{
  vvDebugMsg::msg(1, "vvMovie::vvMovie()");
  scriptName = NULL;
  steps = NULL;
  _canvas = canvas;
  _currentStep = 0;
}

//----------------------------------------------------------------------------
/// Destructor
vvMovie::~vvMovie()
{
  vvDebugMsg::msg(1, "vvMovie::~vvMovie()");
  delete scriptName;
  delete steps;
}

//----------------------------------------------------------------------------
/** Load a movie script from a .vms file.
  @param filename complete path with file name of VMS file
*/
vvMovie::ErrorType vvMovie::load(const char* filename)
{
  vvDebugMsg::msg(1, "vvMovie::load()");

  vvTokenizer* tokenizer;
  ErrorType result=VV_OK;

  delete scriptName;
  scriptName = new char[strlen(filename) + 1];
  strcpy(scriptName, filename);

  std::ifstream file(scriptName);
  if (!file.is_open()) return VV_FILE_ERROR;

  // Initialize stream tokenizer:
  tokenizer = new vvTokenizer(file);
  tokenizer->setCommentCharacter('#');
  tokenizer->setEOLisSignificant(false);
  tokenizer->setCaseConversion(vvTokenizer::VV_LOWER);
  tokenizer->setParseNumbers(true);

  // Read vvMovie steps:
  delete steps;
  steps = new vvSLList<vox::vvMovieStep*>();
  do
  {
    result = parseCommand(tokenizer, steps);
  } while (result==VV_OK);

  switch (result)
  {
    case VV_INVALID_PARAM:
      cerr << "Movie script: invalid command in line " << tokenizer->getLineNumber() << endl;
      break;
    default: break;
  }

  // If last step is not a show, add one for convenience:
  steps->last();
  if (steps->getData()->transform!=SHOW)
  {
    vvMovieStep* step = new vvMovieStep();
    step->transform = SHOW;
    steps->append(step, vvSLNode<vvMovieStep*>::NORMAL_DELETE);
  }

  delete tokenizer;
  if (result==VV_EOF) result = VV_OK;
  return result;
}

//----------------------------------------------------------------------------
/** Parse one command of the movie script.
  @param tokenizer pointer to tokenizer
  @param list      current movie command list used
  @return OK, EOF or INVALID_PARAM
*/
vvMovie::ErrorType vvMovie::parseCommand(vvTokenizer* tokenizer, vvSLList<vvMovieStep*>* list)
{
  vvMovieStep* step;
  ErrorType result;
  TransformType trans = NONE;
  float par[vvMovieStep::MAX_NUM_PARAM];
  int repetitions;
  int i,j;
  vvSLList<vvMovieStep*>* rep = NULL;

  for (i=0; i<vvMovieStep::MAX_NUM_PARAM; ++i)
    par[i] = 0.f;

  if (tokenizer->nextToken() == vvTokenizer::VV_EOF) return VV_EOF;

  if (strcmp(tokenizer->sval, "trans") == 0)
  {
    trans = TRANS;
    if (tokenizer->nextToken() != vvTokenizer::VV_WORD) return VV_INVALID_PARAM;
    if      (strcmp(tokenizer->sval, "x") == 0) par[0] = 0.0f;
    else if (strcmp(tokenizer->sval, "y") == 0) par[0] = 1.0f;
    else if (strcmp(tokenizer->sval, "z") == 0) par[0] = 2.0f;
    else return VV_INVALID_PARAM;
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    par[1] = tokenizer->nval;
  }
  else if (strcmp(tokenizer->sval, "rot") == 0)
  {
    trans = ROT;
    if (tokenizer->nextToken() != vvTokenizer::VV_WORD) return VV_INVALID_PARAM;
    if      (strcmp(tokenizer->sval, "x") == 0) par[0] = 0.0f;
    else if (strcmp(tokenizer->sval, "y") == 0) par[0] = 1.0f;
    else if (strcmp(tokenizer->sval, "z") == 0) par[0] = 2.0f;
    else return VV_INVALID_PARAM;
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    par[1] = tokenizer->nval;
  }
  else if (strcmp(tokenizer->sval, "scale") == 0)
  {
    trans = SCALE;
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    par[0] = tokenizer->nval;
  }
  else if (strcmp(tokenizer->sval, "timestep") == 0)
  {
    trans = TIMESTEP;
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    par[0] = tokenizer->nval;
  }
  else if (strcmp(tokenizer->sval, "nextstep") == 0)
  {
    trans = NEXTSTEP;
  }
  else if (strcmp(tokenizer->sval, "prevstep") == 0)
  {
    trans = PREVSTEP;
  }
  else if (strcmp(tokenizer->sval, "movepeak") == 0)
  {
    trans = MOVEPEAK;
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    par[0] = tokenizer->nval;
  }
  else if (strcmp(tokenizer->sval, "setpeak") == 0)
  {
    trans = SETPEAK;
    for (i=0; i<2; ++i)
    {
      if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
      par[i] = tokenizer->nval;
    }
  }
  else if (strcmp(tokenizer->sval, "setclip") == 0)
  {
    trans = SETCLIP;
    for (i=0; i<4; ++i)
    {
      if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
      par[i] = tokenizer->nval;
    }
  }
  else if (strcmp(tokenizer->sval, "moveclip") == 0)
  {
    trans = MOVECLIP;
    for (i=0; i<4; ++i)
    {
      if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
      par[i] = tokenizer->nval;
    }
  }
  else if (strcmp(tokenizer->sval, "setclipparam") == 0)
  {
    trans = SETCLIPPARAM;
    for (i=0; i<3; ++i)
    {
      if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
      par[i] = tokenizer->nval;
    }
  }
  else if (strcmp(tokenizer->sval, "setquality") == 0)
  {
    trans = SETQUALITY;
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    par[0] = tokenizer->nval;
  }
  else if (strcmp(tokenizer->sval, "changequality") == 0)
  {
    trans = CHANGEQUALITY;
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    par[0] = tokenizer->nval;
  }
  else if (strcmp(tokenizer->sval, "show") == 0)
  {
    trans = SHOW;
  }
  else if (strcmp(tokenizer->sval, "repeat") == 0)
  {
    if (tokenizer->nextToken() != vvTokenizer::VV_NUMBER) return VV_INVALID_PARAM;
    repetitions = int(tokenizer->nval);
    rep = new vvSLList<vvMovieStep*>();
    do
    {
      result = parseCommand(tokenizer, rep);
    } while (result==VV_OK);
    if (result != VV_END_REPEAT)
    {
      rep->removeAll();
      delete rep;
      rep = NULL;
      return VV_INVALID_PARAM;
    }
    // Now attach new list several times to this list:
    for (i=0; i<repetitions; ++i)
    {
      rep->first();
      do
      {
        step = new vvMovieStep();
        for (j=0; j<vvMovieStep::MAX_NUM_PARAM; ++j)
        {
          step->param[j]  = rep->getData()->param[j];
        }
        step->transform = rep->getData()->transform;
        list->append(step, vvSLNode<vvMovieStep*>::NORMAL_DELETE);
      } while (rep->next());
    }
    delete rep;
    return VV_OK;
  }
  else if (strcmp(tokenizer->sval, "endrep") == 0)
  {
    return VV_END_REPEAT;
  }
  else return VV_INVALID_PARAM;

  // Create new movie step list entry:
  step = new vvMovieStep();
  for (j=0; j<vvMovieStep::MAX_NUM_PARAM; ++j)
  {
    step->param[j] = par[j];
  }
  step->transform = trans;
  list->append(step, vvSLNode<vvMovieStep*>::NORMAL_DELETE);
  return VV_OK;
}

//----------------------------------------------------------------------------
/// Returns name of movie script
const char* vvMovie::getScriptName()
{
  vvDebugMsg::msg(1, "vvMovie::getScriptName()");
  return scriptName;
}

//----------------------------------------------------------------------------
/// Returns number of movie script steps
int vvMovie::getNumScriptSteps()
{
  vvDebugMsg::msg(1, "vvMovie::getNumScriptSteps()");

  int cnt = 0;

  if (steps && steps->count()>0)
  {
    steps->first();
    do
    {
      if (steps->getData()->transform == SHOW) ++cnt;
    } while (steps->next());
  }
  return cnt;
}

//----------------------------------------------------------------------------
/** Set a specific movie step.
  @param step index of desired movie step (first step = 0)
  @return true if step was set correctly
*/
bool vvMovie::setStep(size_t step)
{
  vvDebugMsg::msg(1, "vvMovie::setStep()");

  vvMatrix m;
  float axis[3];                                  // rotation axis
  size_t timestep = 0;                            // volume dataset index
  size_t i;
  bool done;
  float peak[2] = {0.f, 0.f};                     // peak position and width
  bool tfChanged = false;

  if (steps==NULL || steps->count()==0) return false;

  _canvas->resetObjectView();
  steps->first();
  for (i=0; i<=step; ++i)
  {
    done = false;
    while (!done)
    {
      switch (steps->getData()->transform)
      {
        default:
        case NONE:
          break;
        case TRANS:
          m.identity();
          axis[0] = axis[1] = axis[2] = 0.0f;
          axis[int(steps->getData()->param[0])] = steps->getData()->param[1];
          m.translate(axis[0], axis[1], axis[2]);
          _canvas->transformObject(m);
          break;
        case ROT:
          m.identity();
          axis[0] = axis[1] = axis[2] = 0.0f;
          axis[int(steps->getData()->param[0])] = 1.0f;
          m.rotate(steps->getData()->param[1] * float(TS_PI) / 180.0f, axis[0], axis[1], axis[2]);
          _canvas->transformObject(m);
          break;
        case SCALE:
          m.identity();
          m.scaleLocal(steps->getData()->param[0]);
          _canvas->transformObject(m);
          break;
        case TIMESTEP:
          timestep = size_t(steps->getData()->param[0]);
          break;
        case NEXTSTEP:
          if (timestep < _canvas->_vd->frames-1)
            ++timestep;
          else timestep = 0;
          break;
        case PREVSTEP:
          if (timestep>0) --timestep;
          else timestep = _canvas->_vd->frames-1;
          break;
        case MOVEPEAK:
          peak[0] += steps->getData()->param[0];
          _canvas->_vd->tf[0].deleteWidgets(vvTFWidget::TF_PYRAMID);
          _canvas->_vd->tf[0].deleteWidgets(vvTFWidget::TF_BELL);
          _canvas->_vd->tf[0]._widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, peak[0], peak[1], 0.0f));
          tfChanged = true;
          break;
        case SETPEAK:
          peak[0] = steps->getData()->param[0];
          peak[1] = steps->getData()->param[1];
          _canvas->_vd->tf[0].deleteWidgets(vvTFWidget::TF_PYRAMID);
          _canvas->_vd->tf[0].deleteWidgets(vvTFWidget::TF_BELL);
          _canvas->_vd->tf[0]._widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, peak[0], peak[1], 0.0f));
          tfChanged = true;
          break;
        case SETCLIP:
          if (steps->getData()->param[0]==0.0f && steps->getData()->param[1]==0.0f &&
              steps->getData()->param[2]==0.0f && steps->getData()->param[3]==0.0f)
          {
            _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_MODE, false);
          }
          else
          {
            _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_MODE, true);
            vec3 normal(steps->getData()->param[0], steps->getData()->param[1], steps->getData()->param[2]);
            normal = normalize(normal);
            vec3 point = normal * steps->getData()->param[3];
            _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PLANE_NORMAL, normal);
            _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PLANE_POINT, point);
          }
          break;
        case MOVECLIP:
        {
          vec3 normal = _canvas->_renderer->getParameter(vvRenderState::VV_CLIP_PLANE_NORMAL);
          vec3 point = _canvas->_renderer->getParameter(vvRenderState::VV_CLIP_PLANE_POINT);
          normal += vec3(steps->getData()->param[0], steps->getData()->param[1], steps->getData()->param[2]);
          normal = normalize(normal);
          vec3 diff = normal * steps->getData()->param[3];
          point += diff;
          _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PLANE_NORMAL, normal);
          _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PLANE_POINT, point);
          break;
        }
        case SETCLIPPARAM:
        {
          _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_SINGLE_SLICE, (steps->getData()->param[0] == 0.0f) ? false : true);
          _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_OPAQUE, (steps->getData()->param[1] == 0.0f) ? false : true);
          _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PERIMETER, (steps->getData()->param[2] == 0.0f) ? false : true);
        }
        break;
        case SETQUALITY:
          _canvas->_renderer->setParameter(vvRenderState::VV_QUALITY, steps->getData()->param[0]);
          break;
        case CHANGEQUALITY:
          _canvas->_renderer->setParameter(vvRenderState::VV_QUALITY,
                  _canvas->_renderer->getParameter(vvRenderState::VV_QUALITY).asFloat() + steps->getData()->param[0]);
          break;
        case SHOW:
          done = true;
          break;
      }
      steps->next();
    }
  }
  if (tfChanged)
  {
    _canvas->_renderer->updateTransferFunction();
    tfChanged = false;
  }
  _canvas->_renderer->setCurrentFrame(timestep);
  _currentStep = step;
  return true;
}

//----------------------------------------------------------------------------
size_t vvMovie::getStep()
{
  return _currentStep;
}

//----------------------------------------------------------------------------
/** Write a movie to disk.
  @param width,height  disk image format
  @param baseFilename  base filename for saved movie frames (w/o extension!)
  @return true if successful
*/
bool vvMovie::write(int width, int height, const char* baseFilename)
{
  vvDebugMsg::msg(1, "vvMovie::write()", width, height);

  vvFileIO* fio;
  vvVolDesc* imgVD;
  char* currentFile;                              // current file name
  uchar* image = NULL;                            // image data
  int i, numSteps;

  if (steps==NULL || width<=0 || height<=0) return false;

  image = new uchar[width * height * 3];          // reserve RGB image space
  currentFile = new char[strlen(baseFilename) + 1 + 20];

  numSteps = getNumScriptSteps();
  for (i=0; i<numSteps; ++i)
  {
    // Render screenshot to memory:
    setStep(i);
    _canvas->draw();
    _canvas->_renderer->renderVolumeRGB(width, height, image);

    // Create filename:
    if (numSteps<100000) sprintf(currentFile, "%s%05d.tif", baseFilename, i);
    else sprintf(currentFile, "%s%09d.tif", baseFilename, i);

    // Write screenshot to file:
    imgVD = new vvVolDesc(currentFile, width, height, image);
    cerr << "Writing file: " << currentFile << endl;
    fio = new vvFileIO();
    fio->saveVolumeData(imgVD, false);
    delete fio;
    delete imgVD;
  }

  delete[] image;
  return true;
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
