/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>

#include "gwApp.h"
#include "gwTier.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

gwTier::gwTier(int i, Typ t, gwApp *a, int g)
{
    id = i;
    group = g;
    app = a;
    type = t;
    origType = t;
    stepNum = 0;
    lifeTime = 100;
    path.reserve(1000);
    direction = ((float)rand() / (float)RAND_MAX) * 2.0 * M_PI;
    oldHabitatValue = 0;
    currentHabitatValue = 0;
    stepInHabitat = 0;
    reorientStep = 0;
    currentState = RandomWalk;
}

void gwTier::setPos(float x, float y)
{
    pos.set(x, y);
    currentHabitatValue = oldHabitatValue = app->getValue(pos);
}

float gwTier::getDir(int x, int y)
{
    vec2 dir = vec2(x * app->pSize[0], y * app->pSize[0]) - pos;
    dir.normalize();
    if (dir[1] > 0)
        return (asin(dir[0]));
    else
        return ((2 * M_PI) - asin(dir[0]));
}

void gwTier::scan()
{
    int startPosX = (int)(pos[0] / app->pSize[0]);
    int startPosY = (int)(pos[1] / app->pSize[1]);
    int numSteps = app->params[type].visionRange / app->pSize[0];
    std::pair<float, float> currentMinMaxAngle;
    std::list<std::pair<float, float> > angles;
    currentMinMaxAngle.first = -1000;
    currentMinMaxAngle.second = -1000;
    float bestValue = 10000;
    for (int r = 1; r <= numSteps; r++)
    {
        int x;
        int y = startPosY + r;
        int value = 300;
        bool checkAngle = false;
        bool wasInsideRange = false;
        currentMinMaxAngle.first = -1000;
        currentMinMaxAngle.second = -1000;
        std::list<std::pair<float, float> >::iterator it = angles.begin();
        if (it != angles.end())
        {
            currentMinMaxAngle = *it;
            checkAngle = true;
        }
        float currentDir;
        for (x = startPosX - r; x <= startPosX + r; x++)
        {
            if (checkAngle)
            {
                currentDir = getDir(x, y);
                while ((currentMinMaxAngle.second > -1000) && (currentDir >= currentMinMaxAngle.second + 0.001) && (it != angles.end()))
                {
                    it++;
                    if (it != angles.end())
                        currentMinMaxAngle = *it;
                }
                if (currentDir < currentMinMaxAngle.second + 0.001 && currentDir > currentMinMaxAngle.first - 0.001)
                {
                    // we are in a direction which is blocked, thus
                    wasInsideRange = true;
                    continue;
                }
                else
                {
                    wasInsideRange = false;
                }
            }
            value = app->getValue(x, y);
            if (value >= app->numHabitatValues - 1)
            { // block this direction
                if (!checkAngle)
                {
                    currentDir = getDir(x, y);
                    checkAngle = true;
                    currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                    angles.push_back(currentMinMaxAngle);
                }
                else
                {
                    if (wasInsideRange) // extend the range
                    {
                        currentMinMaxAngle.second = it->second = currentDir;
                    }
                    else // start new range
                    {
                        currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                        angles.push_back(currentMinMaxAngle);
                        it = angles.end();
                        wasInsideRange = true;
                    }
                }
            }
            if (value < currentHabitatValue)
            {
                if (value < bestValue)
                {
                    bestValue = value;
                    direction = getDir(x, y);
                }
            }
        }
        for (y--; y >= startPosY - r; y--)
        {
            if (checkAngle)
            {
                currentDir = getDir(x, y);
                while ((currentMinMaxAngle.second > -1000) && (currentDir >= currentMinMaxAngle.second + 0.001) && (it != angles.end()))
                {
                    it++;
                    if (it != angles.end())
                        currentMinMaxAngle = *it;
                }
                if (currentDir < currentMinMaxAngle.second + 0.001 && currentDir > currentMinMaxAngle.first - 0.001)
                {
                    // we are in a direction which is blocked, thus
                    wasInsideRange = true;
                    continue;
                }
                else
                {
                    wasInsideRange = false;
                }
            }
            value = app->getValue(x, y);
            if (value >= app->numHabitatValues - 1)
            { // block this direction
                if (!checkAngle)
                {
                    currentDir = getDir(x, y);
                    checkAngle = true;
                    currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                    angles.push_back(currentMinMaxAngle);
                }
                else
                {
                    if (wasInsideRange) // extend the range
                    {
                        currentMinMaxAngle.second = it->second = currentDir;
                    }
                    else // start new range
                    {
                        currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                        angles.push_back(currentMinMaxAngle);
                        it = angles.end();
                        wasInsideRange = true;
                    }
                }
            }
            if (value < currentHabitatValue)
            {
                if (value < bestValue)
                {
                    bestValue = value;
                    direction = getDir(x, y);
                }
            }
        }
        for (x--; x >= startPosX - r; x--)
        {
            if (checkAngle)
            {
                currentDir = getDir(x, y);
                while ((currentMinMaxAngle.second > -1000) && (currentDir >= currentMinMaxAngle.second + 0.001) && (it != angles.end()))
                {
                    it++;
                    if (it != angles.end())
                        currentMinMaxAngle = *it;
                }
                if (currentDir < currentMinMaxAngle.second + 0.001 && currentDir > currentMinMaxAngle.first - 0.001)
                {
                    // we are in a direction which is blocked, thus
                    wasInsideRange = true;
                    continue;
                }
                else
                {
                    wasInsideRange = false;
                }
            }
            value = app->getValue(x, y);
            if (value >= app->numHabitatValues - 1)
            { // block this direction
                if (!checkAngle)
                {
                    currentDir = getDir(x, y);
                    checkAngle = true;
                    currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                    angles.push_back(currentMinMaxAngle);
                }
                else
                {
                    if (wasInsideRange) // extend the range
                    {
                        currentMinMaxAngle.second = it->second = currentDir;
                    }
                    else // start new range
                    {
                        currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                        angles.push_back(currentMinMaxAngle);
                        it = angles.end();
                        wasInsideRange = true;
                    }
                }
            }
            if (value < currentHabitatValue)
            {
                if (value < bestValue)
                {
                    bestValue = value;
                    direction = getDir(x, y);
                }
            }
        }
        for (y++; y <= startPosY + r; y++)
        {
            if (checkAngle)
            {
                currentDir = getDir(x, y);
                while ((currentMinMaxAngle.second > -1000) && (currentDir >= currentMinMaxAngle.second + 0.001) && (it != angles.end()))
                {
                    it++;
                    if (it != angles.end())
                        currentMinMaxAngle = *it;
                }
                if (currentDir < currentMinMaxAngle.second + 0.001 && currentDir > currentMinMaxAngle.first - 0.001)
                {
                    // we are in a direction which is blocked, thus
                    wasInsideRange = true;
                    continue;
                }
                else
                {
                    wasInsideRange = false;
                }
            }
            value = app->getValue(x, y);
            if (value >= app->numHabitatValues - 1)
            { // block this direction
                if (!checkAngle)
                {
                    currentDir = getDir(x, y);
                    checkAngle = true;
                    currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                    angles.push_back(currentMinMaxAngle);
                }
                else
                {
                    if (wasInsideRange) // extend the range
                    {
                        currentMinMaxAngle.second = it->second = currentDir;
                    }
                    else // start new range
                    {
                        currentMinMaxAngle.first = currentMinMaxAngle.second = currentDir;
                        angles.push_back(currentMinMaxAngle);
                        it = angles.end();
                        wasInsideRange = true;
                    }
                }
            }
            if (value < currentHabitatValue)
            {
                if (value < bestValue)
                {
                    bestValue = value;
                    direction = getDir(x, y);
                }
            }
        }
    }
}

int gwTier::checkForNewHabitat(vec2 v)
{
    float len = v.length();
    vec2 vn = v;
    vn.normalize();
    float dist = 0;
    vec2 pn = pos;
    int val = currentHabitatValue;
    while (dist < len)
    {
        dist += app->pSize[0];
        if (dist < len)
        {
            pn = pos + (vn * dist);
        }
        else
        {
            pn = pos + v;
        }
        if (pn[0] >= app->size[0])
            pn[0] = app->size[0] - 0.001;
        if (pn[0] < (float)0.001)
            pn[0] = (float)0.001;
        if (pn[1] >= app->size[1])
            pn[1] = app->size[1] - 0.001;
        if (pn[1] < (float)0.001)
            pn[1] = (float)0.001;
        val = app->getValue(pn);
        if (val != currentHabitatValue && app->params[type].stopAtBoundary[val] == 1) //  we continue or not depending on config file we are crossing a Habitat Should we continue in this habitat as far as we can or should we stop right at the start??
        {
            pos = pn;
            return val;
        }
    }
    pos = pn;
    return val;
}

gwTier::Motion gwTier::move()
{

    if (lifeTime <= 0)
    {
        currentState = Died;
        return currentState;
    }
    if (path.size() > app->params[type].settlement)
    {
        type = Philopatric;
    }
    stepNum++;
    stepInHabitat++;
    reorientStep++;
    int numTries = 0;

    vec2 oldPos;
    oldPos = pos;
    float percentage = 0;

    do // do it multiple times if we can't walk in the direction we wanted but turned around
    {
        percentage = ((float)rand() / (float)RAND_MAX) * 100.0;

        if (currentState == DirectedWalk)
        {
            direction += ((float)rand() / (float)RAND_MAX) * app->params[type].streightness;
            if (stepInHabitat > app->params[type].scanStartFromStep)
            {
                scan();
            }

            if (reorientStep > app->params[type].reorientation)
            {
                direction = ((float)rand() / (float)RAND_MAX) * 2.0 * M_PI;
                reorientStep = 0;
            }
        }
        else // random walk
        {
            direction = ((float)rand() / (float)RAND_MAX) * 2.0 * M_PI;

            if (stepInHabitat > app->params[type].scanStartFromStep)
            {
                scan();
            }
            // switch to DirectedWalk mode?
            if (stepNum > app->params[type].directedWalkFromStep && currentHabitatValue >= app->params[type].directedWalkFromValue)
            {
                currentState = DirectedWalk;
                reorientStep = 0;
            }
        }

        vec2 v;
        int index = app->sigmoDist.size() * ((float)rand() / (float)RAND_MAX);
        if (index >= app->sigmoDist.size())
            index = app->sigmoDist.size() - 1;
        float currentSpeed = (app->sigmoDist[index] / 100.0) * app->params[type].maxSpeed[currentHabitatValue];

        v.set(sin(direction) * currentSpeed, cos(direction) * currentSpeed);

        currentHabitatValue = checkForNewHabitat(v);
        if (currentHabitatValue >= (app->numHabitatValues - 1) || percentage > app->transitionMatrix[oldHabitatValue][currentHabitatValue])
        {
            direction += (float)M_PI; // turn 180 degrees
            pos = oldPos; // stay where you are, don't move into the next habitat
            //if(oldHabitatValue==1 && currentHabitatValue == 3 && type == Colonizers)
            //   fprintf(stderr,"turned around\n");
            currentHabitatValue = oldHabitatValue;
        }

        numTries++;
    } while (oldPos == pos && numTries < 1000);
    if (numTries >= 1000)
    {
        cerr << "oops, this animal did not move anymore, assume it is dead" << endl;
        lifeTime = 0;
    }

    if (currentHabitatValue != oldHabitatValue)
    {
        //if(oldHabitatValue==1 && currentHabitatValue == 3 && type == Colonizers)
        //   fprintf(stderr,"crossed\n");
        stepInHabitat = 0;
        stepNum = 0;
        if (app->params[type].directedWalkFromStep > 0)
        {
            // do a reorientation phase == random walk when entering a new habitat
            currentState = RandomWalk;
        }
    }
    oldHabitatValue = currentHabitatValue;
    lifeTime -= 1;
    percentage = ((float)rand() / (float)RAND_MAX) * 99.99;
    if (currentHabitatValue > app->numHabitatValues || percentage < app->deathRate[currentHabitatValue])
    {
        lifeTime = 0; // die from time to time
    }
    if (path.size() > 0 && pos.x() == path[path.size() - 1].x() && pos.y() == path[path.size() - 1].y())
    {
        cerr << "oops, same position" << endl;
    }
    path.push_back(pos);
    return currentState;
}

gwTier::Motion gwTier::moveOld()
{

    if (lifeTime <= 0)
    {
        currentState = Died;
        return currentState;
    }
    stepNum++;
    stepInHabitat++;
    reorientStep++;
    if (currentState == DirectedWalk)
    {
        direction += ((float)rand() / (float)RAND_MAX) * app->params[type].streightness;
        if (stepInHabitat > app->params[type].scanStartFromStep)
        {
            scan();

            if (reorientStep > app->params[type].reorientation)
            {
                direction = ((float)rand() / (float)RAND_MAX) * 2.0 * M_PI;
                reorientStep = 0;
            }
        }
    }
    else
    {
        direction = ((float)rand() / (float)RAND_MAX) * 2.0 * M_PI;
        // switch to DirectedWalk mode?
        if (stepNum > app->params[type].directedWalkFromStep && currentHabitatValue >= app->params[type].directedWalkFromValue)
        {
            currentState = DirectedWalk;
            reorientStep = 0;
        }
    }

    vec2 v;
    int index = app->sigmoDist.size() * ((float)rand() / (float)RAND_MAX);
    if (index >= app->sigmoDist.size())
        index = app->sigmoDist.size() - 1;
    float currentSpeed = (app->sigmoDist[index] / 100.0) * app->params[type].maxSpeed[currentHabitatValue];

    v.set(sin(direction) * currentSpeed, cos(direction) * currentSpeed);
    vec2 oldPos;
    oldPos = pos;
    pos += v;
    if (pos[0] >= app->size[0])
        pos[0] = app->size[0] - 0.001;
    if (pos[0] < (float)0.001)
        pos[0] = (float)0.001;
    if (pos[1] >= app->size[1])
        pos[1] = app->size[1] - 0.001;
    if (pos[1] < (float)0.001)
        pos[1] = (float)0.001;

    currentHabitatValue = app->getValue(pos);
    float percentage = ((float)rand() / (float)RAND_MAX) * 100.0;
    if (currentHabitatValue >= app->numHabitatValues - 1 || percentage >= app->transitionMatrix[oldHabitatValue][currentHabitatValue])
    {
        direction += (float)M_PI; // turn 180 degrees
        pos = oldPos; // stay where you are, don't move into the next habitat
    }
    else if (currentHabitatValue != oldHabitatValue)
    {
        stepInHabitat = 0;
        if (currentState == DirectedWalk && (app->stopEveryTransition || (currentHabitatValue < oldHabitatValue)) && currentHabitatValue <= app->params[type].satisfactoryHabitat)
        {
            // we are so happy, we reached a better habitat which is satisfying enough, thus we stay here for a while
            stepNum = 0;
            currentState = RandomWalk;
        }
    }
    oldHabitatValue = currentHabitatValue;
    lifeTime -= 1;
    percentage = ((float)rand() / (float)RAND_MAX) * 99.99;
    if (currentHabitatValue > app->numHabitatValues || percentage < app->deathRate[currentHabitatValue])
    {
        lifeTime = 0; // die from time to time
    }
    if (pos.x() == path[path.size() - 1].x() && pos.y() == path[path.size() - 1].y())
    {
        cerr << "oops, same position 2" << endl;
    }
    path.push_back(pos);
    return currentState;
}

void gwTier::writeSVG(FILE *fp)
{
    if (path.size() > 1)
    {

        fprintf(fp, "<path d=\"M%d,%d ", (int)(path[0][0] / app->pSize[0]), (int)(path[0][1] / app->pSize[1]));

        for (int i = 1; i < path.size(); i++)
        {
            fprintf(fp, " L%d,%d", (int)(path[i][0] / app->pSize[0]), (int)(path[i][1] / app->pSize[1]));
        }

        if (type == Philopatric)
        {
            fprintf(fp, "\" fill=\"none\" stroke=\"#888888\" stroke-width=\"2\" />\n");
        }
        else
        {
            fprintf(fp, "\" fill=\"none\" stroke=\"#880011\" stroke-width=\"2\" />\n");
        }
        fprintf(fp, "<rect x=\"%d\" y=\"%d\" width=\"2\" height=\"2\" fill=\"blue\" stroke=\"none\" stroke-width=\"0\"/>\n", (int)(path[path.size() - 1][0] / app->pSize[0]), (int)(path[path.size() - 1][1] / app->pSize[1]));
    }
}

void gwTier::writeShape(FILE *fp)
{
    int numSteps = path.size();
    char buf[10000];
    if (numSteps > 0)
    {
        for (int i = 0; i < numSteps; i++)
        {
            if (i == (numSteps - 1))
                sprintf(buf, "%d\t%d\t%f\t%f\t1\t%d\t%d\n", id, i, path[i][0] + app->XLLCorner, path[i][1] + app->YLLCorner, (int)type, (int)origType);
            else
                sprintf(buf, "%d\t%d\t%f\t%f\t0\t%d\t%d\n", id, i, path[i][0] + app->XLLCorner, path[i][1] + app->YLLCorner, (int)type, (int)origType);
            char *c = buf;
            while (*c != '\0')
            {
                if (*c == '.')
                    *c = ',';
                c++;
            }
            fputs(buf, fp);
        }
    }
}
