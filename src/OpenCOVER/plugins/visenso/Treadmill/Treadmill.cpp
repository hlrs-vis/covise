/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Treadmill.h"

#define REPORT(x)                                                        \
    std::cout << __FILE__ << ":" << __LINE__ << ": " << #x << std::endl; \
    x;

#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Geometry>
#include <osg/Image>
#include <osgDB/ReadFile>

#include <cover/coVRFileManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>

// 45.0 is somehow a constant to get the right 1:1 mapping from/to quake3 and cover
const double QUAKE_COVER_SCALE = 45.0;
const double FORWARD_YAW_RATIO = 2.6;

Treadmill *Treadmill::plugin = NULL;

Treadmill::Treadmill()
: coVRPlugin(COVER_PLUGIN_NAME)
{
#ifdef _WIN32
    _dirSeparator = "\\";
#else
    _dirSeparator = "/";
#endif

    _forwardSpeed = 0;
    _yawSpeed = 0;
    _sideSpeed = 0;
    _turnDecision = -1;

    _port = (int)coCoviseConfig::getInt("value", "COVER.Plugin.Treadmill.Port", 5555);
    _timeout = (int)coCoviseConfig::getInt("value", "COVER.Plugin.Treadmill.Timeout", 3600);

    _forwardSpeedScale = (double)coCoviseConfig::getFloat("value", "COVER.Plugin.Treadmill.ForwardSpeedScale", 0.05);
    _yawSpeedScale = (double)coCoviseConfig::getFloat("value", "COVER.Plugin.Treadmill.YawSpeedScale", _forwardSpeedScale / FORWARD_YAW_RATIO); // 0.0192);
    _sideSpeedScale = (double)coCoviseConfig::getFloat("value", "COVER.Plugin.Treadmill.SideSpeedScale", 0.03);
    _floorOffset = (double)coCoviseConfig::getFloat("value", "COVER.Plugin.Treadmill.FloorOffset", 60.0);

    _levelsDirectory = coCoviseConfig::getEntry("value", "COVER.Plugin.Treadmill.LevelsDirectory", "/work/common/Projekte/RBK/quake_levels/");

    _animalPopup = new TexturePopup(0.0, 768.0, 100.0, 100.0);
    _arrowLeftPopup = new TexturePopup(0.0, 80.0, 80.0, 80.0);
    _arrowRightPopup = new TexturePopup(1024.0 - 80.0, 80.0, 80.0, 80.0);

    _arrowLeftPopup->setImageFile(_levelsDirectory + _dirSeparator + "textures" + _dirSeparator + "arrows" + _dirSeparator + "arrow_left.jpg");
    _arrowRightPopup->setImageFile(_levelsDirectory + _dirSeparator + "textures" + _dirSeparator + "arrows" + _dirSeparator + "arrow_right.jpg");
}

Treadmill::~Treadmill()
{
    if (coVRMSController::instance()->isMaster())
    {
        delete _messageReceiver;
    }
}

double to_double(std::string s)
{
    double d;
    std::istringstream iss(s);
    iss >> d;
    return d;
}

template <class T>
void operator>>(const std::string &s, T &converted)
{
    std::istringstream iss(s);
    iss >> converted;
    if ((iss.rdstate() & std::istringstream::failbit) != 0)
    {
        std::cerr << "Error in conversion from string \""
                  << s
                  << "\" to type "
                  << typeid(T).name()
                  << std::endl;
    }
}

bool Treadmill::init()
{
    if (plugin)
        return false;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nTreadmillPlugin::TreadmillPlugin\n");

    Treadmill::plugin = this;

    if (coVRMSController::instance()->isMaster())
    {
        //std::cerr << "Creating MessageReceiver..." << std::endl;
        _messageReceiver = new MessageReceiver(_port, _timeout);
    }

    return true;
}

void Treadmill::preFrame()
{
    //   char buffer[100];
    //   buffer[100] = buffer[0] = 0;
    static int delay = 0;
    if (coVRMSController::instance()->isMaster() && (delay++ % 1 == 0))
    {
        std::vector<std::string> queue = _messageReceiver->popMessageQueue();

        for (size_t i = 0; i < queue.size(); i++)
        {
            //std::cout << "queue[" << i << "] = " << queue[i] << std::endl;

            // break up message into tokens
            std::vector<std::string> tokens;
            std::string token;
            std::istringstream iss(queue[i]);
            while (std::getline(iss, token, ' '))
            {
                tokens.push_back(token);
            }

            _handleTokens(tokens);
        }

        // drive/walk depending on forward/yaw speed

        /* calc delta vector for translation (hand position relative to click point) */
        double sF = VRSceneGraph::instance()->scaleFactor();
        osg::Vec3 delta(_sideSpeed * (sF / QUAKE_COVER_SCALE), _forwardSpeed * (sF / QUAKE_COVER_SCALE), 0.0); //startHandPos-handPos;

        /* get xform matrix, translate to make viewPos the origin */
        osg::Matrix dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
        osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
        dcs_mat.postMult(osg::Matrix::translate(-viewerPos[0], -viewerPos[1], -viewerPos[2]));

        /* apply translation */
        dcs_mat.postMult(osg::Matrix::translate(frameFactor(delta[0]),
                                                frameFactor(delta[1]),
                                                frameFactor(delta[2])));

        /* apply direction change */
        osg::Matrix rot_mat;
        osg::Vec3 dirAxis(0.0, 0.0, 1.0);
        if ((dirAxis[0] != 0.0) || (dirAxis[1] != 0.0) || (dirAxis[2] != 0.0))
        {
            rot_mat.makeRotate(frameFactor(_yawSpeed) * M_PI / 180, dirAxis[0], dirAxis[1], dirAxis[2]);
            dcs_mat.mult(dcs_mat, rot_mat);
        }

        /* undo handPos translation, set new xform matrix */
        osg::Matrix tmp;
        tmp.makeTranslate(viewerPos[0], viewerPos[1], viewerPos[2]);
        dcs_mat.postMult(tmp);
        VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);

        // the master just changed its view, sync the slaves
        //   coVRCollaboration::instance()->SyncXform();

        // read L/R buttons for direction choice
        _handleMouseButtons();
    }
    else
    {
        // TODO: slave reads message from master when hud-image changes

        //char buffer[100];
        //buffer[100] = buffer[0] = 0;

        //coVRMSController::instance()->readMaster((char *)&buffer[0], 100);
    }

/*
   // only the master receives control messages from space gui
   if(coVRMSController::instance()->isMaster())
   {
      fd_set sready;
      struct timeval nowait;
      
      FD_ZERO(&sready);
      FD_SET(_serverSocket->get_id(), &sready);
      memset((char *)&nowait,0,sizeof(nowait));
      
      if ( select(_serverSocket->get_id()+1, &sready, NULL, NULL, &nowait) == 0)
      {
         // nothing to see here, move along
      }
      else
      {
         // read package
         
         int numRead = 0;
         numRead = _serverSocket->Read(buffer, 99);
         if (numRead != -1)
         {
            buffer[numRead] = 0;
            
            // send a short ack to indicate we are ready
            char ack[1];
            ack[0] = 255;
            _serverSocket->write(ack, 1);
         }
      }
      
      // send the buffer to slaves
      coVRMSController::instance()->sendSlaves((char *)&buffer[0], 100);
   }
   else  // slave
   {
      // read buffer from master
      // perhaps only the 1st byte to know if theres something important. if it is 0, dont read anything more!!!
      coVRMSController::instance()->readMaster((char *)&buffer[0], 100);
   }


   if (buffer[0] != 0)
   {
      // break up message into tokens
      std::vector<std::string> tokens;
      std::string token;
      std::istringstream iss(buffer);
      while (std::getline(iss, token, ' '))
      {
         tokens.push_back(token);
      }

      _handleTokens(tokens);
   }
*/

// do some info printing
#if 0
   osg::Vec3 dcs_trans;
   osg::Quat dcs_rot;
   osg::Vec3 dcs_scale;
   osg::Quat dcs_so;
   dcs_mat.decompose(dcs_trans, dcs_rot, dcs_scale, dcs_so);
   double angle, x, y, z;
   dcs_rot.getRotate(angle, x, y, z);
   //std::cout << "angle, x, y, z = " << angle << ", " << x << ", " << y << ", " << z << std::endl;
   std::cout << "dcs_mat->getTrans() = " << dcs_trans[0] << ", " << dcs_trans[1] << ", " << dcs_trans[2] << std::endl;
   
   
   osg::Vec3 mpi_trans;
   double mpi_angle, mpi_x, mpi_y;
   mpi_angle = -angle*(180.0/M_PI) + 90.0;
   //std::cout << "mpi_angle = " << mpi_angle << std::endl;
   dcs_mat.postMult(osg::Matrix::translate(-viewerPos[0], -viewerPos[1], -viewerPos[2]));
   dcs_mat.postMultRotate(dcs_rot.inverse());
   mpi_trans = dcs_mat.getTrans();
   mpi_trans = -mpi_trans;
   //std::cout << "mpi_trans = " << mpi_trans[0] << ", " << mpi_trans[1] << ", " << mpi_trans[2] << std::endl;
   std::cout << "scaleFactor = " << VRSceneGraph::instance()->scaleFactor() << std::endl;
   std::cout << "cover->getScale() = " << cover->getScale() << std::endl;
#endif
}

void Treadmill::_handleMouseButtons()
{
    if (cover->getMouseButton()->wasPressed())
    {
        int state = cover->getMouseButton()->getState();
        //std::cout << "cover->getMouseButton()->getState() = " << state << std::endl;
        switch (state)
        {
        case 1:
            _turnDecision = 0;
            std::cout << "_turnDecision = " << _turnDecision << std::endl;
            break;
        case 112:
            _turnDecision = 1;
            std::cout << "_turnDecision = " << _turnDecision << std::endl;
            break;
        default:
            break;
        }
    }
}

void Treadmill::_handleTokens(const std::vector<std::string> &tokens)
{
    // if received command was "getValues" then send player position
    if (tokens[0] == "getValues")
    {
        //if(coVRMSController::instance()->isMaster())
        {
            osg::Matrix dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
            osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
            osg::Vec3 dcs_trans;
            osg::Quat dcs_rot;
            osg::Vec3 dcs_scale;
            osg::Quat dcs_so;
            dcs_mat.decompose(dcs_trans, dcs_rot, dcs_scale, dcs_so);
            double angle, x, y, z;
            dcs_rot.getRotate(angle, x, y, z);

            // only yaw is of interest, flip it when z-axis shows down instead of up
            if (z < 0.0)
            {
                angle = -angle;
            }

            osg::Vec3 mpi_trans;
            double mpi_angle, mpi_x, mpi_y;
            mpi_angle = -angle * (180.0 / M_PI) + 90.0;
            //std::cout << "mpi_angle = " << mpi_angle << std::endl;
            dcs_mat.postMult(osg::Matrix::translate(-viewerPos[0], -viewerPos[1], -viewerPos[2]));
            dcs_mat.postMultRotate(dcs_rot.inverse());
            mpi_trans = dcs_mat.getTrans();
            mpi_trans = -mpi_trans;
            //std::cout << "mpi_trans = " << mpi_trans[0] << ", " << mpi_trans[1] << ", " << mpi_trans[2] << std::endl;
            double sF = VRSceneGraph::instance()->scaleFactor();
            mpi_x = mpi_trans[0] * (QUAKE_COVER_SCALE / sF);
            mpi_y = mpi_trans[1] * (QUAKE_COVER_SCALE / sF);

            // 50 is hardcoded in MPI software
            char out[50 + 1];
            out[50] = 0; // make it c string
            for (int i = 0; i < 50; i++)
            {
                out[i] = ' ';
            }
            sprintf(out, "%lf %lf %lf 0 0 %d |", mpi_x, mpi_y, mpi_angle, _turnDecision);
            // _serverSocket->write(out, 50);
            //_messageReceiver->write(out, 50);
            std::string outString(out);
            outString.resize(50, ' ');
            _messageReceiver->send(outString);

            if (_turnDecision != -1)
            {
                // we just sent the turn decision to space gui, so reset it
                _turnDecision = -1;
                std::cout << "_turnDecision = " << _turnDecision << std::endl;
            }
        }
    }
    else if (tokens[0] == "demo")
    {
        if (tokens[1] == "0")
        {
            coVRFileManager::instance()->unloadFile();
        }
        else if (tokens[1] == "1")
        {
            _turnDecision = -1;
            _forwardSpeed = 0.0;
            _sideSpeed = 0.0;
            _yawSpeed = 0.0;

            _animalPopup->hide();
            _arrowLeftPopup->hide();
            _arrowRightPopup->hide();

            std::string fileName = _levelsDirectory + _dirSeparator + std::string("demo.wrl");
            coVRFileManager::instance()->loadFile(fileName.c_str());
        }
    }
    else if (tokens[0] == "map")
    {
        coVRFileManager::instance()->unloadFile();

        _animalPopup->hide();
        _arrowLeftPopup->hide();
        _arrowRightPopup->hide();

        if (tokens[1] == "familiarization1" || tokens[1] == "familiarization2" || tokens[1] == "trainingCN1" || tokens[1] == "trainingCN2" || tokens[1] == "training1" || tokens[1] == "controlled1" || tokens[1] == "controlled2")
        {
            std::string fileName = _levelsDirectory + tokens[1] + std::string(".wrl");
            coVRFileManager::instance()->loadFile(fileName.c_str());
        }
        else
        {
            std::cout << "Unknown or unsupported level: " << tokens[1] << std::endl;
        }
    }
    else if (tokens[0] == "forwardSpeed")
    {
        // change walk/drive speed
        std::cout << "forwardSpeed = " << tokens[1] << std::endl;
        int speed;
        tokens[1] >> speed;
        _forwardSpeed = -speed * _forwardSpeedScale;
    }
    else if (tokens[0] == "yawSpeed")
    {
        // change yaw speed
        std::cout << "yawSpeed = " << tokens[1] << std::endl;
        int speed;
        tokens[1] >> speed;
        _yawSpeed = -speed * _yawSpeedScale;
    }
    else if (tokens[0] == "sideSpeed")
    {
        std::cout << "sideSpeed = " << tokens[1] << std::endl;
        int speed;
        tokens[1] >> speed;
        _sideSpeed = -speed * _sideSpeedScale;
    }
    else if (tokens[0] == "leftArrow")
    {
        std::cout << "leftArrow " << tokens[1] << std::endl;
        int showArrow;
        tokens[1] >> showArrow;
        if (showArrow)
        {
            _arrowLeftPopup->show();
        }
        else
        {
            _arrowLeftPopup->hide();
        }
    }
    else if (tokens[0] == "rightArrow")
    {
        std::cout << "rightArrow " << tokens[1] << std::endl;
        int showArrow;
        tokens[1] >> showArrow;
        if (showArrow)
        {
            _arrowRightPopup->show();
        }
        else
        {
            _arrowRightPopup->hide();
        }
    }
    else if (tokens[0] == "animal")
    {
        for (size_t i = 0; i < tokens.size(); i++)
        {
            std::cout << tokens[i] << " ";
        }
        std::cout << std::endl;

        if (tokens[1] == "Door") // special case for door!
        {
            std::string filename = _levelsDirectory + _dirSeparator + "textures" + _dirSeparator + "doors" + _dirSeparator + "door1.jpg";
            std::cout << "Loading door from file " << filename << std::endl;
            _animalPopup->setImageFile(filename);
            _animalPopup->show();
        }
        else
        {
            std::string filename = _levelsDirectory + _dirSeparator + "textures" + _dirSeparator + tokens[1] + ".jpg";
            std::cout << "Loading animal from file " << filename << std::endl;
            _animalPopup->setImageFile(filename);
            _animalPopup->show();
        }
    }
    else if (tokens[0] == "playerPosition")
    {
        for (size_t i = 0; i < tokens.size(); i++)
        {
            std::cout << tokens[i] << " ";
        }
        std::cout << std::endl;

        double x, y, angle;
        tokens[1] >> x;
        tokens[2] >> y;
        tokens[3] >> angle; // in degree!

        // set player position -> move/rotate world
        osg::Matrix current_transform = osg::Matrix::identity(); // = VRSceneGraph::instance()->getTransform()->getMatrix();
        osg::Vec3 current_viewer = cover->getViewerMat().getTrans();

        osg::Vec3 axis(0.0, 0.0, 1.0);
        osg::Quat q(-(angle - 90.0) * (M_PI / 180.0), axis);
        osg::Quat q_id(0.0, axis);
        osg::Matrix q_mat;
        q.get(q_mat);

        double sF = VRSceneGraph::instance()->scaleFactor();
        current_transform.setTrans(-x * (sF / QUAKE_COVER_SCALE), -y * (sF / QUAKE_COVER_SCALE), -_floorOffset * (sF / QUAKE_COVER_SCALE));
        current_transform.postMult(q_mat);

        current_transform.postMult(osg::Matrix::translate(current_viewer[0], current_viewer[1], current_viewer[2]));

        VRSceneGraph::instance()->getTransform()->setMatrix(current_transform);

        //      coVRCollaboration::instance()->SyncXform();
    }
    else if (tokens[0] == "trialFeedback")
    {
        std::cout << "trialFeedback = " << tokens[1] << std::endl;
        int trialTime; // in minutes
        tokens[1] >> trialTime;
        if (trialTime == -1)
        {
            // clear message popup
        }
        else
        {
            // show message popup of elapsed time
        }
    }
    else if (tokens[0] == "feedback")
    {
        std::cout << "feedback = " << tokens[1] << std::endl;
        int foundTargets;
        tokens[1] >> foundTargets;
        if (foundTargets == -1)
        {
            // clear message popup
        }
        else
        {
            // show message popup of num found targets
        }
    }
    else if (tokens[0] == "exit")
    {
        coVRFileManager::instance()->unloadFile();

        _animalPopup->hide();
        _arrowLeftPopup->hide();
        _arrowRightPopup->hide();

        _turnDecision = -1;
    }
    else
    {
        std::cout << "Unknown command: ";
        std::vector<std::string>::const_iterator it;
        for (it = tokens.begin(); it != tokens.end(); ++it)
        {
            std::cout << *it << " ";
        }
        std::cout << std::endl;
    }
}

double Treadmill::frameFactor(double delta)
{
    return delta * cover->frameDuration() * 60.0; // somehow adjust to 60fps
}

void Treadmill::key(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == 'a' || keySym == osgGA::GUIEventAdapter::KEY_Page_Up)
        {
            // request left turn
            _turnDecision = 0;
            std::cout << "_turnDecision = " << _turnDecision << std::endl;
        }
        else if (keySym == 'd' || keySym == osgGA::GUIEventAdapter::KEY_Page_Down)
        {
            // request right turn
            _turnDecision = 1;
            std::cout << "_turnDecision = " << _turnDecision << std::endl;
        }
    }
}

COVERPLUGIN(Treadmill)
