/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GridStep.h"

#include "Rect.h"
#include "Strct.h"
#include "USG.h"

// supported Gridtypes (should be...) :
//    UNIGRD
//    RCTGRD
//    STRGRD
//    CELGRD
//    UNSGRD

// initialize the structure with the given objects and saveSearch/needGradient
GridStep::GridStep(const coDistributedObject *g, const coDistributedObject *v, int sS, int bBa, float attA, int nG, int nR, Grid *parent, int s)
{
    //coDistributedObject **gSetIn = NULL, **vSetIn = NULL;

    int i;

    // reset/init
    numBlocks = 0;
    block = NULL;

    grid = parent;
    myStep = s;

    stepDuration = NULL;
    numSteps = 1;

    blockCache = -1;

    needGradient = nG;
    needRot = nR;

    // maybe we got the special case with static grid and changing data and
    //  we are not the first GridStep
    if (!g && !v && s > 0)
    {
        cerr << "#+#  GridStep::GridStep --- generating fake-blocks" << endl;

        // we then have to get the data from the first step
        numBlocks = parent->getStep(0)->getNumBlocks();
        block = new GridBlock *[numBlocks + 1];
        for (i = 0; i < numBlocks; i++)
        {
            // hmmm.......
            block[i] = grid->getStep(0)->block[i];

            //block[i] = new GridBlock(this, grid, i);
        }
        // done
        return;
    }

    // check what we got
    const char *dataType;
    dataType = g->getType();
    if (!strcmp(dataType, "SETELE"))
    {
        //cerr << "GridStep::GridStep   multiblock initializing" << endl;

        // multiblock
        // check for special case
        if (v->getAttribute("TIMESTEP"))
        {
            // static multiblock grid with changing data

            cerr << "GridStep::GridStep   static multiblock grid with changing data not yet implemented" << endl;

            // we have to resort the data-structure from   TIMESTEPs->MULTIBLOCK
            //     to   MULTIBLOCK->TIMESTEPs  so it is conforming to our structure
        }
        else
        {
            cerr << "* GridStep::GridStep   multiblock  initializing" << endl;

            // multiblock grid with data (both static)
            const coDistributedObject *const *setGridIn, *const *setDataIn;
            int numSetGrid, numSetData;

            // check integrity/get objects
            setGridIn = ((coDoSet *)g)->getAllElements(&numSetGrid);
            setDataIn = ((coDoSet *)v)->getAllElements(&numSetData);
            if (numSetGrid != numSetData)
            {
                cerr << "### GridStep::GridStep  ---  inconsitent numbers of grid-/data-elements in set" << endl;

                return;
            }

            // continue initialization
            numBlocks = numSetGrid;
            block = new GridBlock *[numBlocks + 1];
            stepDuration = new float[numBlocks + 1];
            for (i = 0; i < numBlocks; i++)
            {

                dataType = setGridIn[i]->getType();
                if (!strcmp(dataType, "RCTGRD"))
                    block[i] = new RectBlock(this, grid, i);
                else if (!strcmp(dataType, "STRGRD"))
                    block[i] = new StrctBlock(this, grid, i);
                else if (!strcmp(dataType, "UNSGRD"))
                    block[i] = new USGBlock(this, grid, i);
                else if (!strcmp(dataType, "UNIGRD"))
                    block[i] = new RectBlock(this, grid, i);
                else
                    block[i] = NULL; // ERROR

                // TODO            cerr << "   NOTE: not checking for any attributes that might supply a stepDuration" << endl;

                stepDuration[i] = 0.0;
                const char *stepDurationAttr = setGridIn[i]->getAttribute("STEPDURATION");
                if (stepDurationAttr)
                {
                    sscanf(stepDurationAttr, "%f", &stepDuration[i]);
                }
                const char *rotSpeed = setGridIn[i]->getAttribute("ROT_SPEED");
                const char *rotAxis = setGridIn[i]->getAttribute("ROT_AXIS");
                if (rotSpeed && rotAxis)
                {
                    float speed, rax, ray, raz;
                    sscanf(rotSpeed, "%f", &speed);
                    sscanf(rotAxis, "%f %f %f", &rax, &ray, &raz);
                    block[i]->setRotation(speed, rax, ray, raz);
                }

                if (!block[i]->initialize(setGridIn[i], setDataIn[i], sS, bBa, attA))
                {
                    delete block[i];
                    block[i] = NULL;
                }
            }
            block[i] = NULL;
            stepDuration[i] = 0.0;

            cerr << "* GridStep::GridStep  ---  " << numBlocks << " blocks initialized" << endl;

            /*

         // usual multiblock grid/data
         gSetIn = ((coDoSet *)g)->getAllElements( &numBlocks );
         vSetIn = ((coDoSet *)v)->getAllElements( &i );
         dataType = gSetIn[0]->getType();
         if( i!=numBlocks )
         return;   // ERROR !!!!
         block = new GridBlock*[numBlocks+1];
         stepDuration = new float[numBlocks+1];
         for( i=0; i<numBlocks; i++ )
         {
         if( !strcmp(dataType, "RCTGRD") )
         block[i] = new RectBlock( this, grid );
         else
         block[i] = NULL;   // ERROR :)
         stepDuration[i] = 0.0;
         block[i]->initialize(gSetIn[i], vSetIn[i], sS, bBa, attA);
         }
         block[i] = NULL;
         stepDuration[i] = 0.0;
         */
        }
    }
    else
    {
        // single block
        dataType = v->getType();
        if (!strcmp(dataType, "SETELE"))
        {
            cerr << "* GridStep::GridStep   singleblock+changing data   initializing" << endl;

            // static singleblock grid with changing data
            numBlocks = 1;
            block = new GridBlock *[2];
            //coDistributedObject *const *tObj = ((coDoSet *)v)->getAllElements( &numSteps );
            ((coDoSet *)v)->getAllElements(&numSteps);
            stepDuration = new float[numSteps + 1];
            dataType = g->getType();
            if (!strcmp(dataType, "RCTGRD"))
                block[0] = new RectBlock(this, grid, 0);
            else if (!strcmp(dataType, "STRGRD"))
                block[0] = new StrctBlock(this, grid, 0);
            else if (!strcmp(dataType, "UNSGRD"))
                block[0] = new USGBlock(this, grid, 0);
            else if (!strcmp(dataType, "UNIGRD"))
                block[0] = new RectBlock(this, grid, i);
            else
                block[0] = NULL; // ERROR

            cerr << "   NOTE: not checking for any attributes that might supply a stepDuration" << endl;

            for (i = 0; i < numSteps; i++)
                stepDuration[i] = 0.0;

            if (!block[0]->initialize(g, v, sS, bBa, attA))
            {
                delete block[0];
                block[0] = NULL;
            }
            block[1] = NULL;
            stepDuration[numSteps] = 0.0;
        }
        else
        {
            cerr << "* GridStep::GridStep   singleblock  initializing" << endl;

            // usual singleblock grid/data
            numBlocks = 1;
            block = new GridBlock *[2];
            stepDuration = new float[2];
            dataType = g->getType();
            if (!strcmp(dataType, "RCTGRD"))
                block[0] = new RectBlock(this, grid, 0);
            else if (!strcmp(dataType, "STRGRD"))
                block[0] = new StrctBlock(this, grid, 0);
            else if (!strcmp(dataType, "UNSGRD"))
                block[0] = new USGBlock(this, grid, 0);
            else if (!strcmp(dataType, "UNIGRD"))
                block[0] = new RectBlock(this, grid, i);
            else
                block[0] = NULL; // ERROR

            cerr << "   NOTE: not checking for any attributes that might supply a stepDuration" << endl;

            stepDuration[0] = 0.0;

            if (!block[0]->initialize(g, v, sS, bBa, attA))
            {
                cerr << "   WARNING: GridBlock->initialize failed" << endl;

                delete block[0];
                block[0] = NULL;
            }
            block[1] = NULL;
            stepDuration[1] = 0.0;
        }
    }

    // prepare cache
    if (numBlocks)
    {
        fromBlockCache = new int *[numBlocks];
        for (i = 0; i < numBlocks; i++)
        {
            fromBlockCache[i] = new int[numBlocks + 1];
            fromBlockCache[i][0] = -1;
        }

        fromBlockCache2 = new int *[numBlocks];
        for (i = 0; i < numBlocks; i++)
        {
            fromBlockCache2[i] = new int[numBlocks + 1];
            fromBlockCache2[i][0] = -1;
        }
    }
    else
    {
        fromBlockCache = NULL;
        fromBlockCache2 = NULL;
    }

    // done
    return;
}

GridStep::~GridStep()
{
    int i;
    if (numSteps)
    {
        delete[] block;
    }
    else
    {
        if (numBlocks && block)
        {
            for (i = 0; i < numBlocks; i++)
                if (block[i])
                    delete block[i];
            delete[] block;
        }
    }
    if (stepDuration)
        delete[] stepDuration;

    if (numBlocks && fromBlockCache)
    {
        for (i = 0; i < numBlocks; i++)
            delete[] fromBlockCache[i];
        delete[] fromBlockCache;
    }

    return;
}

float GridStep::getStepDuration(int s)
{
    if (!stepDuration || s >= numSteps)
        return (0.0);
    return (stepDuration[s]);
}

int GridStep::getNumBlocks()
{
    return (numBlocks);
}

int GridStep::getNumSteps()
{
    return (numSteps);
}

Particle *GridStep::startParticle(float x, float y, float z, int lsbCache)
{
    int iBlock = -1;
    GridBlockInfo *gbi = NULL;

    // lsbCache specifies in which block of the previous timestep this
    //  particle was (if available...otherwise -1)

    // first go through all cached blocks
    if (lsbCache >= 0)
    {
        int i;
        for (i = 0; fromBlockCache[lsbCache][i] != -1; i++)
        {
            if (block[fromBlockCache[lsbCache][i]])
            {
                if (block[fromBlockCache[lsbCache][i]]->initGBI(x, y, z, gbi))
                {
                    //cerr << "*** ls cache hit ***" << endl;
                    return (this->startParticle(gbi));
                }
            }
        }
    }

    // es koennte noch eingebaut werden, dass die bereits im cache durchsuchten
    //  bloecke nachher bei einer neusuche nicht mehr durchsucht werden...mal sehen
    //  vorerst mal nur den cache implementieren und sehen was das bringt

    /* das war nur zum debuggen...hat aber nichts geholfen -> block->initGBI weiter suchen
   if( block[49]->initGBI(x, y, z, gbi) )
   {
      cerr << "found in block 49" << endl;
      return( this->startParticle(gbi) );
   }
   else
      cerr << "checked block 49 in step " << myStep << " for " << x << " " << y << " " << z << "  , but not found!" << endl;
   */

    //   cerr << "[ GridStep::startParticle ]" << endl;

    // we have to determine in which block this particle starts

    // did we cache something ? (only if no lsbCache specified as that is much more accurate)
    if (blockCache != -1 && lsbCache == -1)
    {
        //            cerr << "cache try" << endl;

        // cache exists so TRY
        if (block[blockCache])
        {
            if (!block[blockCache]->initGBI(x, y, z, gbi))
            {
                // MISS
                //                        cerr << "cache miss  (blockCache: " << blockCache << " , " << x << " " << y << " " << z << ")" << endl;
                iBlock = blockCache;
            }
            else
            {
                // HIT
                //                        cerr << "cache hit" << endl;
                addFromBlockCache(lsbCache, gbi->blockNo);
                return (this->startParticle(gbi));
            }
        }
    }

    // nothing cached or cache MISS

    // TEST
    //if( initGBI(x, y, z, gbi, iBlock) )
    if (initGBI(x, y, z, gbi, -1, -1))
    {
        cerr << "particle found (we are step: " << myStep << ", block: " << gbi->block->blockNo() << ")" << endl;

        addFromBlockCache(lsbCache, gbi->blockNo);

        // particle is inside the grid...
        return (this->startParticle(gbi));
    }

    // not found
    return (NULL);
}

void GridStep::addFromBlockCache(int from, int to)
{
    if (from < 0)
        return;
    int i;

    //cerr << "adding cache: step: " << myStep << "  from: " << from << "   to: " << to << endl;

    for (i = 0; fromBlockCache[from][i] != -1 && fromBlockCache[from][i] != to; i++)
        ;
    if (fromBlockCache[from][i] != -1)
        return;
    fromBlockCache[from][i] = to;
    fromBlockCache[from][i + 1] = -1;
    return;
}

void GridStep::addFromBlockCache2(int from, int to)
{
    if (from < 0)
        return;
    int i;

    //cerr << "adding cache2: step: " << myStep << "  from: " << from << "   to: " << to << endl;

    for (i = 0; fromBlockCache2[from][i] != -1 && fromBlockCache2[from][i] != to; i++)
        ;
    if (fromBlockCache2[from][i] != -1)
        return;
    fromBlockCache2[from][i] = to;
    fromBlockCache2[from][i + 1] = -1;
    return;
}

Particle *GridStep::startParticle(GridBlockInfo *sourceGBI)
{
    if (needGradient)
        grid->getGradient(sourceGBI);
    if (needRot)
        grid->getRotation(sourceGBI);

    return (new Particle(grid, sourceGBI, NULL, needGradient, needRot));
}

int GridStep::initGBI(float x, float y, float z, GridBlockInfo *&gbi, int iBlock, int fromBlock)
{
    int b;
    //   GridBlockInfo newGBI;

    //cerr << "[ GridStep::initGBI ]" << endl;

    // try cached blocks first
    if (fromBlock >= 0)
    {
        for (b = 0; fromBlockCache2[fromBlock][b] != -1; b++)
        {
            if (block[fromBlockCache2[fromBlock][b]]->initGBI(x, y, z, gbi))
            {
                // HIT
                //cerr << "*** HIT ***" << endl;
                return (1);
            }
        }

        //cerr << "*** failed after: " << b << "  tries" << endl;
    }

    // did we cache something ?
    if (blockCache != -1 && blockCache != iBlock)
    {
        //cerr << "cache try" << endl;

        if (block[blockCache])
        {
            // TRY
            if (block[blockCache]->initGBI(x, y, z, gbi))
            {
                // HIT
                addFromBlockCache2(fromBlock, b);
                return (1);
            }
        }
    }

    //cerr << "numBlocks=" << numBlocks << "     iBlock=" << iBlock << "    blockCache=" << blockCache << endl;

    // nothing cached or cache MISS
    // loop over all blocks (except iBlock) in this step
    for (b = 0; b < numBlocks; b++)
    {
        if (!block[b])
            cerr << "WARNING:  block[b=" << b << "] is NULL" << endl;

        if (b != iBlock && block[b])
        {
            if (block[b]->initGBI(x, y, z, gbi))
            {
                //            cerr << "GridStep::initGBI: found particle" << endl;
                if (!gbi)
                    cerr << "gbi is NULL" << endl;

                // update cache
                blockCache = b;

                addFromBlockCache2(fromBlock, b);

                return (1);
            }
        }
    }

    return (0);
}
