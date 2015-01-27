/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Grid.h"

#include <util/coviseCompat.h>

Grid::Grid(const coDistributedObject *g, const coDistributedObject *v, int sS, int bBa, float attA, int nG, int nR)
{
    const char *dataType;
    const coDistributedObject *const *gSetIn, *const *vSetIn;

    int i;

    // reset
    numSteps = 0;
    steps = NULL;
    stepDuration = NULL;
    sourceGBI = NULL;
    numSources = 0;
    maxSources = 0;
    dataChangesFlag = 0;
    sbcFlag = 0;

    // see what we got
    dataType = g->getType();
    if (!strcmp(dataType, "SETELE")) // special case, single block , ignore block structure but just use the single element MULTIBLOC-TIMESTEPS does not work yet
    { // also treats a single timestep as stationary.
        gSetIn = ((coDoSet *)g)->getAllElements(&numSteps);
        vSetIn = ((coDoSet *)v)->getAllElements(&i);
        if ((numSteps == 1) && (i == 1))
        {
            g = gSetIn[0];
            v = vSetIn[0];
        }
        numSteps = 0;
        dataType = g->getType();
    }
    float stepDur = 0.0;
    const char *stepDurationAttr = g->getAttribute("STEPDURATION");
    if (stepDurationAttr)
    {
        sscanf(stepDurationAttr, "%f", &stepDur);
    }

    if (!strcmp(dataType, "SETELE"))
    {
        //cerr << "### Grid::Grid  ---  dataType  SETELE   not yet supported" << endl;

        // the grid is a set
        //  -> check if multiblock or timesteps
        if (g->getAttribute("TIMESTEP"))
        {
            //cerr << "### Grid::Grid  ---  dataType  SETELE(TIMESTEP)   not yet supported" << endl;
            cerr << "### Grid:Grid  ---  TIMESTEP Grid detected" << endl;
            gSetIn = ((coDoSet *)g)->getAllElements(&numSteps);
            vSetIn = ((coDoSet *)v)->getAllElements(&i);
            if (i != numSteps)
            {
                cerr << "### ERROR: number of steps from grid and data aren't equal" << endl;
                return;
            }

            steps = new GridStep *[numSteps + 1];
            stepDuration = new float[numSteps + 1];
            for (i = 0; i < numSteps; i++)
            {
                steps[i] = new GridStep(gSetIn[i], vSetIn[i], sS, bBa, attA, nG, nR, this, i);
                stepDuration[i] = steps[i]->getStepDuration();
            }
            steps[i] = NULL;
            stepDuration[i] = 0.0;

            cerr << "### Grid::Grid  ---   " << numSteps << " steps initialized" << endl;

            /*
         // it might well be timestep+multiblock, but that will be handled by GridStep
         gSetIn = ((coDoSet *)g)->getAllElements( &numSteps );
         vSetIn = ((coDoSet *)v)->getAllElements( &i );
         if( i!=numSteps )
            return;   // ERROR !!!!
         steps = new GridStep*[numSteps+1];
         stepDuration = new float[numSteps+1];
         for( i=0; i<numSteps; i++ )
         {
            steps[i] = new GridStep(gSetIn[i], vSetIn[i], sS, bBa, attA, 0, 0, this);
         stepDuration[i] = steps[i]->getStepDuration();
         }
         // this is just to be sure....shouldn't be needed, but to be on the save side :)
         steps[i] = NULL;
         stepDuration[i] = 0.0;
         */
        }
        else
        {
            // only multiblock, but still might have time-dependant data
            if (v->getAttribute("TIMESTEP"))
            {
                cerr << "### Grid::Grid  ---  dataType  SETELE with DATA-TIMESTEP   TODO: check destructor, reuse grid insead of a simple copy" << endl;
                vSetIn = ((coDoSet *)v)->getAllElements(&numSteps);

                steps = new GridStep *[numSteps + 1];
                stepDuration = new float[numSteps + 1];
                for (i = 0; i < numSteps; i++)
                {
                    steps[i] = new GridStep(g, vSetIn[i], sS, bBa, attA, nG, nR, this, i);
                    stepDuration[i] = steps[i]->getStepDuration();
                    if (stepDuration[i] == 0.0)
                        stepDuration[i] = stepDur;
                }
                steps[i] = NULL;
                stepDuration[i] = 0.0;
            }
            else
            {
                cerr << "### Grid::Grid  ---  multiblock Grid detected" << endl;

                // multiblock grid with data (both static)
                numSteps = 1;
                steps = new GridStep *[2];
                stepDuration = new float[2];
                steps[0] = new GridStep(g, v, sS, bBa, attA, nG, nR, this, 0);
                steps[1] = NULL;
                stepDuration[0] = steps[0]->getStepDuration();
                stepDuration[1] = 0.0;
            }
        }
    }
    else
    {
        cerr << "### Grid::Grid  ---  singleblock Grid detected" << endl;

        // not a set (singleblock)
        // we still might have time-depending data though
        dataType = v->getType();
        if (!strcmp(dataType, "SETELE"))
        {
            cerr << "### Grid::Grid  ---   data has dataType  SETELE" << endl;

            // static grid with time-dependant data
            //coDistributedObject *const *tObj = ((coDoSet *)v)->getAllElements( &numSteps );
            ((coDoSet *)v)->getAllElements(&numSteps);
            steps = new GridStep *[numSteps + 1];
            stepDuration = new float[numSteps + 1];
            steps[0] = new GridStep(g, v, sS, bBa, attA, nG, nR, this, 0);
            stepDuration[0] = steps[0]->getStepDuration();
            for (i = 1; i < numSteps; i++)
            {
                //steps[i] = steps[0];
                steps[i] = new GridStep(NULL, NULL, sS, bBa, attA, nG, nR, this, i);
                stepDuration[i] = steps[0]->getStepDuration(i);
                if (stepDuration[i] == 0.0)
                    stepDuration[i] = stepDur;
            }
            steps[i] = NULL;
            stepDuration[i] = 0.0;
            dataChangesFlag = 1;

            cerr << "### Grid::Grid  ---   " << numSteps << " steps initialized" << endl;
        }
        else
        {
            // simple as can be, just one grid with data
            numSteps = 1;
            steps = new GridStep *[2];
            stepDuration = new float[2];
            steps[0] = new GridStep(g, v, sS, bBa, attA, nG, nR, this, 0);
            steps[1] = NULL;
            stepDuration[0] = steps[0]->getStepDuration();
            stepDuration[1] = 0.0;
        }
    }

    // done
    return;
}

Grid::~Grid()
{
    int i;
    /*  changed on 07.05.2002, each steps[] is a separate object now, even if dataChangesFlag is true
   if( dataChangesFlag )
   {
      delete steps[0];
      delete[] steps;
      steps = NULL;
   } */
    if (steps)
    {
        for (i = 0; i < numSteps; i++)
            if (steps[i])
                delete steps[i];
        delete[] steps;
    }
    if (stepDuration)
        delete[] stepDuration;
    if (sourceGBI)
    {
        for (i = 0; i < numSources; i++)
            if (sourceGBI[i])
                delete sourceGBI[i];
        delete[] sourceGBI;
    }
    return;
}

float Grid::getStepDuration(int s)
{
    if (!stepDuration)
        return (0.0);
    return (stepDuration[s]);
}

int Grid::getNumSteps()
{
    return (numSteps);
}

Particle *Grid::startParticle(int s, float x, float y, float z)
{
    Particle *r = NULL, *tmp = NULL;

    //   cerr << "** Grid::startParticle( " << s << " , " << x << " , " << y << " , " << z << " ) " << endl;

    r = steps[s]->startParticle(x, y, z);
    if (numSteps > 1 && r)
    {
        // initialize Particle->gbiNext
        //  we therfore simply try to start "another" particle in the next step
        //  and if successfull, we merge the both GBIs into one "single" particle

        // a special case is, if we have a static grid with changing data which thus
        //  would mean, that we only have fakeSteps and fakeBlocks (only the first step
        //  and its blocks are "real"), then this initialization can be speed up a lot
        if (dataChangesFlag)
        {
            r->gbiNext = r->gbiCur->createCopy();
            // and update it's step (stepNo / step)
            if (s == numSteps - 1)
            {
                r->gbiNext->stepNo = 0;
                r->gbiNext->step = steps[0];
            }
            else
            {
                r->gbiNext->stepNo = s + 1;
                r->gbiNext->step = steps[s + 1];
            }
            // done
        }
        else
        {
            if (s == numSteps - 1)
                tmp = steps[0]->startParticle(x, y, z);
            else
                tmp = steps[s + 1]->startParticle(x, y, z);

            if (!tmp)
            {
                // we definitely need the particle in the 2 first timesteps...as we don't have it
                //  we assume this particle can't be started

                cerr << "### Grid::startParticle --- particle could only be found in the 1st step but not in the 2nd. Deleting." << endl;

                delete r;
                r = NULL;
            }
            else
            {
                // merge the GBIs
                r->gbiNext = tmp->gbiCur->createCopy();
                delete tmp;
            }
        }
    }

    return (r);
}

int Grid::getGBINextStep(GridBlockInfo *gbiCur, GridBlockInfo *&gbiNext)
{
    gbiNext = NULL;

    int s = gbiCur->stepNo;
    Particle *tmp = NULL;

    //   cerr << "## Grid::getGBINextStep --- poor implementation. can be improved a lot. [" << s << " of " << numSteps << "]" << endl;

    if (s == numSteps - 1)
    {
        tmp = steps[0]->startParticle(gbiCur->x, gbiCur->y, gbiCur->z, gbiCur->blockNo);
        s = 0;
    }
    else
    {
        tmp = steps[s + 1]->startParticle(gbiCur->x, gbiCur->y, gbiCur->z, gbiCur->blockNo);
        s++;
    }

    //   cerr << "getGBINextStep: " << s << "  found: " << (tmp?"yes":"no") << endl;

    if (tmp)
    {
        gbiNext = tmp->gbiCur->createCopy();
        gbiNext->stepNo = s; // required for static grid with changing data
        gbiNext->sbcIdx = gbiCur->sbcIdx;

        //cerr << "?!?" << endl;

        //      cerr << "  in Block: " << gbiNext->blockNo << endl;

        delete tmp;
    }
    else
    {
        // we might have to check SBC here too !!! (not only in getVelocity)
        float x, y, z;

        //      cerr << "getGBINextStep: trying with SBC  (" << gbiCur->sbcIdx+1 << ")" << endl;

        x = gbiCur->x;
        y = gbiCur->y;
        z = gbiCur->z;
        sbcApply(&x, &y, &z);
        tmp = steps[s]->startParticle(x, y, z, gbiCur->blockNo);
        if (tmp)
        {

            /*

         PROBLEM:
         wenn hier sbcIdx nicht hochgesetzt wird, und in Particle.cpp gbiNext->sbcIdx=gbiCur->sbcIdx
         gesetzt wird, dann wird der Partikel der bei getVelocity SBC macht richtig dargestellt, aber
         der bei getGBINextStep SBC macht wird falsch dargestellt.

         woran liegt das ? wenn hier sbcIdx hochgesetzt wird und in Particle.cpp die Zuweisung entfaellt,
         dann ist das Ergebnis gerade andersrum. Seltsam.
         Donnerstag pruefen.

         restliche rausfliegende Partikel fliegen evtl. wegen fehlendem SBC im advanceGBI raus...pruefen

         ok...SBC muss in beide Richtungen probiert werden (da ja beide Raender symmetrisch sind)...
         ...dazu zusaetzlich einen sbcCache einrichten, der die jeweiligen Zielbloecke beinhaltet
         (auch hier beide Richtungen beachten)
         ausserdem muss Ausgabe richtig gemacht werden...sbcIdx wird dabei wieder dekrementiert aber
         irgendwie funktioniert es trotzdem nicht

         bei startparticle immer den letzten gbi mituebergeben, so kann von startpunkt zu
         startpunkt getraced werden oder wenn im naechsten zeitschritt dann kann an der gleichen
         position wie im vorherigen zeitschritt gestartet werden. sollte VIEL bringen !!!

         */
            //         cerr << "getGBINextStep: succeeded with SBC  (" << gbiCur->sbcIdx+1 << ")";

            gbiNext = tmp->gbiCur->createCopy();
            gbiNext->stepNo = s;
            gbiNext->sbcIdx = gbiCur->sbcIdx + 1;
            //gbiNext->sbcIdx++;

            // hmm ok maybe we have to adjust the gbiCur->sbcIdx too....let's try
            //gbiCur->sbcIdx++;

            //         cerr << "  gbiNext-SBC: " << gbiNext->sbcIdx << endl;
        }
        else
        {
            x = gbiCur->x;
            y = gbiCur->y;
            z = gbiCur->z;
            sbcInvApply(&x, &y, &z);
            tmp = steps[s]->startParticle(x, y, z, gbiCur->blockNo);
            if (tmp)
            {
                //            cerr << "getGBINextStep: succeeded with SBC  (" << gbiCur->sbcIdx-1 << ")";

                gbiNext = tmp->gbiCur->createCopy();
                gbiNext->stepNo = s;
                gbiNext->sbcIdx = gbiCur->sbcIdx - 1;
                //gbiNext->sbcIdx++;

                // hmm ok maybe we have to adjust the gbiCur->sbcIdx too....let's try
                //gbiCur->sbcIdx++;

                //            cerr << "  gbiNext-SBC: " << gbiNext->sbcIdx << endl;
            }
        }

        //gbiCur->block->debugTetra(gbiCur->x, gbiCur->y, gbiCur->z, 0.03);
    }

    return (gbiNext ? 1 : 0);
}

Particle *Grid::startParticle(int sID)
{
    if (sourceGBI[sID])
    {

        // TODO: initialize Particle->gbiNext
        cerr << "### Grid::startParticle with a source not yet supports timesteps...." << endl;

        return ((sourceGBI[sID]->step)->startParticle(sourceGBI[sID]));
    }
    return (NULL);
}

int Grid::placeSource(int s, float x, float y, float z)
{
    cerr << "### Grid::placeSource  ---   not yet implemented" << endl;
    cerr << "  s: " << s << "  x/y/z: " << x << " " << y << " " << z << endl;

    /*
      sourceGBI[numSources] = new GridBlockInfo();
      if( steps[s]->initGBI(x, y, z, sourceGBI[numSources]) )
      {
         numSources++;
         return( numSources-1 );
      }
      delete sourceGBI[numSources];
   */
    return (-1);
}

void Grid::moveSource(int sID, float dx, float dy, float dz)
{
    cerr << "### Grid::moveSource  ---   not yet implemented" << endl;
    cerr << "  sID: " << sID << "  dx/dy/dz: " << dx << " " << dy << " " << dz << endl;

    // still to be implemented...
}

/// =============================================================================================
/// =============================================================================================
/// =============================================================================================
/// =============================================================================================
/// =============================================================================================

#include "USG.h"

int Grid::getVelocity(GridBlockInfo *&gbi, float dx, float dy, float dz,
                      float *u, float *v, float *w, float &dc, int sbc0, GridBlockInfo *gbiCache, GridBlockInfo *gbi0,
                      int updateGBI)
{
    int r = gbi->block->getVelocity(gbi, dx, dy, dz, u, v, w, dc);
    if (r == 1) // everything ok
    {
        //   gbi->block->debugTetra(gbi->x+dx, gbi->y+dy, gbi->z+dz, 0.01);
        // remember to also apply SBC to the velocity

        //sbcVel(u,v,w,gbi->sbcIdx);
        gbi->u = *u;
        gbi->v = *v;
        gbi->w = *w;

        return (1);
    }

    //gbi->block->debugTetra(gbi->x+dx, gbi->y+dy, gbi->z+dz, 0.04);

    int s = gbi->sbcIdx;

    //cerr << "gbi: " << s << endl;

    // TODO: sbc macht probleme mit dem cache und irgendwo vermutlich noch transformationsfehler

    // CACHE
    if (gbiCache)
    {
        if (gbiCache->block->getVelocity(gbiCache, dx, dy, dz, u, v, w, dc) == 1)
        {

            //         cerr << "Grid::getVelocity ### cache hit" << endl;
            gbi->u = *u;
            gbi->v = *v;
            gbi->w = *w;

            //cerr << "gbi: " << gbi->sbcIdx << "  cache: " << gbiCache->sbcIdx << endl;

            int sd = gbi->sbcIdx - gbiCache->sbcIdx;

            //cerr << "   sd: " << sd << endl;

            gbi->sbcIdx = gbiCache->sbcIdx;
            while (sd > 0)
            {
                sbcInvApply(&gbi->x, &gbi->y, &gbi->z);
                sd--;
            }
            while (sd < 0)
            {
                sbcApply(&gbi->x, &gbi->y, &gbi->z);
                sd++;
            }

            //gbi->x = gbiCache->x;
            //gbi->y = gbiCache->y;
            //gbi->z = gbiCache->z;

            if (updateGBI)
            {
                //cerr << "! update" << endl;
                *gbi = *gbiCache;
                // cerr << "  gbi: " << gbi->sbcIdx << "  cache: " << gbiCache->sbcIdx << endl;
            }

            return (1);
        }
        //      else
        //         cerr << "Grid::getVelocity ### cache miss    (block " << gbiCache->blockNo << ")" << endl;
    }
    //   else
    //      cerr << "NO CACHE" << endl;

    if (gbi0)
    {
        // we could/should(?) check if gbiCache==gbi0 to avoid double computation....

        if (gbi0->block->getVelocity(gbi0, dx, dy, dz, u, v, w, dc) == 1)
        {

            //         cerr << "Grid::getVelocity ### gbi0 hit" << endl;
            gbi->u = *u;
            gbi->v = *v;
            gbi->w = *w;
            //gbi->sbcIdx = gbi0->sbcIdx;

            int sd = gbi->sbcIdx - gbi0->sbcIdx;
            gbi->sbcIdx = gbi0->sbcIdx;
            while (sd > 0)
            {
                sbcInvApply(&gbi->x, &gbi->y, &gbi->z);
                sd--;
            }
            while (sd < 0)
            {
                sbcApply(&gbi->x, &gbi->y, &gbi->z);
                sd++;
            }
            if (updateGBI)
                *gbi = *gbi0;

            return (1);
        }
        //      else
        //         cerr << "Grid::getVelocity ### gbi0 miss    (block " << gbi0->blockNo << ")" << endl;
    }
    //   else
    //      cerr << "NO GBI0" << endl;

    // attA-condition(2) or out-of-bounds(0)
    //   so at first try to find the NEW position inside another block
    GridBlockInfo *myGBI = NULL;
    if (!gbi->step->initGBI(gbi->x + dx, gbi->y + dy, gbi->z + dz, myGBI, gbi->blockNo, gbi->blockNo))
    {
        // not found, maybe we have symmetric boundaries
        if (sbcFlag)
        {
            //         cerr << ">-----------------------------" << endl;
            //         cerr << "Grid::getVelocity => SBC" << endl;

            /*
         if( gbiSBC && 0 )
         {
           if( s-gbiSBC->sbcIdx == 1 )
           {
              sbcInvApply(&dx, &dy, &dz);
              if( gbiSBC->block->getVelocity( gbiSBC, dx, dy, dz, u, v, w, dc )==1 )
              {
                 cerr << "gbiSBCCache(-1)...hit" << endl;

                 cerr << "   cached sbcIdx: " << gbiSBC->sbcIdx << endl;
         */
            /*
          *gbi = *gbiSBC;

          *u = gbi->u;
          *v = gbi->v;
          *w = gbi->w;
         //sbcVel(u,v,w,gbi->sbcIdx);
         if( sbc0 )
            sbcInvApply(u,v,w);
         gbi->u = *u;
         gbi->v = *v;
         gbi->w = *w;

         return( 1 );*/
            /*
         }
         else
         cerr << "gbiSBCCache...miss" << endl;
         }
         else
         if( s-gbiSBC->sbcIdx == -1 )
         {
         sbcApply(&dx, &dy, &dz);
         if( gbiSBC->block->getVelocity( gbiSBC, dx, dy, dz, u, v, w, dc )==1 )
         {
         cerr << "gbiSBCCache(+1)...hit" << endl;

         cerr << "   cached sbcIdx: " << gbiSBC->sbcIdx << endl;

         }
         else
         cerr << "gbiSBCCache...miss" << endl;
         }
         }*/

            // use symmetric boundary conditions (if given)
            gbi->sbcIdx++;
            s = gbi->sbcIdx;

            // hmm has this to be done or not ? I think its ok to do it.
            //  yes this must be done !!!
            gbi->x += dx;
            gbi->y += dy;
            gbi->z += dz;

            // transform the particle
            //      cerr << "::Grid getVelocity: using SBC   (" << s << ")" << endl;
            //      cerr << "   old pos: " << gbi->x << " " << gbi->y << " " << gbi->z << endl;
            //      cerr << "   sbcIdx : " << gbi->sbcIdx << endl;

            //gbi->block->debugTetra(gbi->x, gbi->y, gbi->z, 0.02);

            sbcApply(&gbi->x, &gbi->y, &gbi->z);

            //      cerr << "   new pos: " << gbi->x << " " << gbi->y << " " << gbi->z << endl;

            // find it's new position/velocity
            if (gbi->step->initGBI(gbi->x, gbi->y, gbi->z, myGBI))
            {
                //         cerr << "found" << endl;

                //gbi->block->debugTetra(gbi->x, gbi->y, gbi->z, 0.06);

                // found
                delete gbi;
                gbi = myGBI;
                gbi->sbcIdx = s;

                //            cerr << "+" << endl;

                //                     cerr << "   new sbcIdx: " << gbi->sbcIdx << endl;

                *u = gbi->u;
                *v = gbi->v;
                *w = gbi->w;
                if (sbc0)
                {
                    //cerr << "+ sbc0" << endl;
                    sbcApply(u, v, w);
                }
                //sbcVel(u,v,w,gbi->sbcIdx);
                gbi->u = *u;
                gbi->v = *v;
                gbi->w = *w;
                //sbcInvApply(&dx, &dy, &dz);
                sbcApply(&dx, &dy, &dz);

                //sbcInvApply( &gbi->x, &gbi->y, &gbi->z );

                r = 1;
            }
            else
            {
                sbcInvApply(&gbi->x, &gbi->y, &gbi->z);
                sbcInvApply(&gbi->x, &gbi->y, &gbi->z);

                //            cerr << "-" << endl;

                if (gbi->step->initGBI(gbi->x, gbi->y, gbi->z, myGBI))
                {
                    //            cerr << "found" << endl;

                    //gbi->block->debugTetra(gbi->x, gbi->y, gbi->z, 0.06);

                    // found
                    delete gbi;
                    gbi = myGBI;
                    gbi->sbcIdx = s - 2;

                    // test, 5.2.03
                    //gbi->sbcIdx = s-1;   sind andere faelle

                    //                        cerr << "   new sbcIdx: " << gbi->sbcIdx << endl;

                    *u = gbi->u;
                    *v = gbi->v;
                    *w = gbi->w;
                    //sbcVel(u,v,w,gbi->sbcIdx);
                    if (sbc0)
                    {
                        //cerr << "- sbc0" << endl;
                        sbcInvApply(u, v, w);
                    }
                    gbi->u = *u;
                    gbi->v = *v;
                    gbi->w = *w;

                    //sbcApply(&dx, &dy, &dz);
                    sbcInvApply(&dx, &dy, &dz);

                    r = 1;
                }
                else
                {

                    //gbi->block->debugTetra(gbi->x, gbi->y, gbi->z, 0.06);
                    gbi->sbcIdx--;
                    //                        cerr << "not found !" << endl;
                }
                //sbcApply( &gbi->x, &gbi->y, &gbi->z );
            }
            //      else
            //         cerr << "not found !" << endl;

            // "virtually" move the particle back to its last position
            gbi->x -= dx;
            gbi->y -= dy;
            gbi->z -= dz;
        }

        if (!r)
        {
            // if not found, try to find it in the current block (maybe it is deformed that much)
            if (gbi->block->initGBI(gbi->x + dx, gbi->y + dy, gbi->z + dz, myGBI))
            {
                // found the point inside the current block...

                // .
                // . TODO
                // .

                //cerr << "Grid::getVelocity ### inside the current block (TODO)" << endl;
                delete gbi;
                gbi = myGBI;
                gbi->sbcIdx = s;

                gbi->x -= dx;
                gbi->y -= dy;
                gbi->z -= dz;

                r = gbi->block->getVelocity(gbi, dx, dy, dz, u, v, w, dc);

                //sbcVel(u,v,w,gbi->sbcIdx);
                gbi->u = *u;
                gbi->v = *v;
                gbi->w = *w;

                cerr << "doch noch gefunden in cell " << ((USGBlockInfo *)gbi)->cell << endl;
            }
            // we'll also end up here if the attA-condition is fullfilled
        }

        if (r)
        {
            float p0[3], p1[3];

            p0[0] = gbi->x;
            p0[1] = gbi->y;
            p0[2] = gbi->z;

            p1[0] = p0[0] + gbi->u;
            p1[1] = p0[1] + gbi->v;
            p1[2] = p0[2] + gbi->w;

            gbi->block->debugTetra(p0, p0, p0, p1);
        }
    }
    else
    {
        // found in another block

        // .
        // . TODO
        // .

        //cerr << "Grid::getVelocity ### moved on to another block (TODO)" << endl;

        //cerr << "Grid::getVelocity ### moved on to another block (block " << myGBI->blockNo << ")...have to test the code if it works properly" << endl;
        //      cerr << "Grid::getVelocity ### moved on to another block (block " << myGBI->blockNo << ") (from " << gbi->blockNo << ")" << endl;

        //gbi->block->debugTetra(gbi->x, gbi->y, gbi->z, 0.01);
        myGBI->block->debugTetra(myGBI->x, myGBI->y, myGBI->z, 0.01f);

        delete gbi;
        gbi = myGBI;
        gbi->sbcIdx = s;

        // we have to use a trick here...maybe it's just a hack, we'll have to see
        gbi->x -= dx;
        gbi->y -= dy;
        gbi->z -= dz;

        r = gbi->block->getVelocity(gbi, dx, dy, dz, u, v, w, dc);
        if (r != 1)
            cerr << "Grid::getVelocity ### this shouldn't happen !!! check Grid.cpp" << endl;

        //sbcVel(u,v,w,gbi->sbcIdx);
        gbi->u = *u;
        gbi->v = *v;
        gbi->w = *w;

        return (1);
    }

    // if still not found, use attA-condition if possible or assume that the particle has
    //   left the grid entirely
    //else
    //   if(!r)
    //      gbi->block->debugTetra(gbi->x+dx, gbi->y+dy, gbi->z+dz, 0.02);

    return (r);

    /*
      int c;
      GridStep *s;
      GridBlockInfo t;
      float tx, ty, tz;

      c = gbi->block->getVelocity( gbi, dx, dy, dz, u, v, w );
      if( c==1 )
      {
         // ok
         return( 1 );
   }

   // try to find that point in another block
   s = gbi->step;
   tx = gbi->x+dx;
   ty = gbi->y+dy;
   tz = gbi->z+dz;

   if( s->initGBI(tx, ty, tz, &t, gbi->blockNo) )
   return( 1 ); // ok

   // couldn't find it in another block -> return original result
   return( c );
   */
}

void Grid::getGradient(GridBlockInfo *gbi)
{
    //gbi->block->getGradient( gbi );
    if (gbi)
        cerr << "Grid::getGradient" << endl;
    return;
}

void Grid::getRotation(GridBlockInfo *gbi)
{
    //gbiNext = tmp->gbiCur->createCopy();

    if (gbi)
        cerr << "Grid::getRotation" << endl;
    //gbi->block->getRotation( gbi );
    return;
}

int Grid::advanceGBI(GridBlockInfo *&gbiCur, GridBlockInfo *&gbiNext,
                     float dx, float dy, float dz, GridBlockInfo *gbiCache, GridBlockInfo *gbiNextCache)
{
    int r;

    //   cerr << "Grid::advanceGBI>>>" << endl;

    if (gbiCur->block)
        r = this->advanceGBI(gbiCur, dx, dy, dz, gbiCache);
    if (gbiNext)
        if (gbiNext->block)
            if (!this->advanceGBI(gbiNext, dx, dy, dz, gbiNextCache))
            {
                //cerr << "HMMM!" << endl;
                //gbiCur->block->debugTetra(gbiCur->x, gbiCur->y, gbiCur->z, 0.3);
                r = 0;
            }
    /*
   if( gbiCur->block )
      r = gbiCur->block->advanceGBI(gbiCur, dx, dy, dz);

   if( gbiNext )
   {
      if( gbiNext->block )
         if( !gbiNext->block->advanceGBI(gbiNext, dx, dy, dz) )
         {
            cerr << "::Grid : advanceGBI(next)  left grid" << endl;
            gbiCur->block->debugTetra(gbiCur->x, gbiCur->y, gbiCur->z, 0.2);
   }
   }
   */

    if (!r)
    {
        // hmm.... PROBLEM

        //cerr << "::Grid : advanceGBI  left grid" << endl;
        //gbiCur->block->debugTetra(gbiCur->x+dx, gbiCur->y+dy, gbiCur->z+dz, 0.1);
    }

    //   cerr << "::Grid : advanceGBI will return " << r << endl;

    //   cerr << "<<<Grid::advanceGBI" << endl;

    return (r);
}

int Grid::advanceGBI(GridBlockInfo *&gbi, float dx, float dy, float dz, GridBlockInfo *gbiCache)
{
    int r;
    float dc;
    //   cerr << "adv------>" << endl;
    r = this->getVelocity(gbi, dx, dy, dz, &gbi->u, &gbi->v, &gbi->w, dc, 0, gbiCache, NULL, 1);
    //   cerr << "<-adv" << endl;
    if (r)
    {
        r = 1;
        gbi->block->advanceGBI(gbi, dx, dy, dz);
    }

    /* 10.02.2004: unnoetig, die Funktion kann in den Abgeleiteten Klassen entfallen
      int r = gbi->block->advanceGBI(gbi, dx, dy, dz);
      int s = gbi->sbcIdx+1;                         // ?!? doesn't seem to matter....is overwritten somewhere i guess
      GridBlockInfo *myGBI = NULL;

      if( !r && sbcFlag )
      {
         sbcApply( &gbi->x, &gbi->y, &gbi->z );
         if( gbi->step->initGBI( gbi->x, gbi->y, gbi->z, myGBI ) )
         {

   //cerr << "advanceGBI: found using SBC (+> " << s << ")" << endl;

   //gbi->block->debugTetra(gbi->x, gbi->y, gbi->z, 0.02);

   // found
   delete gbi;
   gbi = myGBI;
   gbi->sbcIdx = s;

   r = 1;
   }
   else
   {
   sbcInvApply( &gbi->x, &gbi->y, &gbi->z );
   sbcInvApply( &gbi->x, &gbi->y, &gbi->z );
   if( gbi->step->initGBI( gbi->x, gbi->y, gbi->z, myGBI ) )
   {
   //cerr << "advanceGBI: found using SBC (-> " << s-2 << ")" << endl;

   //gbi->block->debugTetra(gbi->x, gbi->y, gbi->z, 0.02);

   // found
   delete gbi;
   gbi = myGBI;
   gbi->sbcIdx = s-2;

   r = 1;
   }
   }
   }
   */

    return (r);
}

GridStep *Grid::getStep(int s)
{
    if (s >= 0 && s < numSteps)
        return (steps[s]);

    cerr << "#*# ERROR #*#   Grid::getStep called with invalid parameter (" << s << ")" << endl;

    return (NULL);
}

int Grid::initGBI(int s, float x, float y, float z, GridBlockInfo *gbi)
{
    //cerr << "***  Grid::initGBI" << endl;

    return (steps[s]->initGBI(x, y, z, gbi));
}

void Grid::useSBC(float axxis[3], float angle)
{

    // problem(?):  how do we handle it if gbiCur is still inside the grid, but gbiNext not and
    //    gbiNext passes the SBC-border...hmmm

    cerr << "Grid::useSBC => initializing SBC" << endl;

    float c, s, v;
    c = (float)cos((angle * M_PI) / 180.0);
    s = (float)sin((angle * M_PI) / 180.0);
    v = 1.0f - c;

    sbcFlag = 1;

    // [row][col]
    sbcMatrix[0][0] = axxis[0] * axxis[0] * v + c;
    sbcMatrix[0][1] = axxis[0] * axxis[1] * c - axxis[2] * s;
    sbcMatrix[0][2] = axxis[0] * axxis[2] * v + axxis[1] * s;

    sbcMatrix[1][0] = axxis[0] * axxis[1] * v + axxis[2] * s;
    sbcMatrix[1][1] = axxis[1] * axxis[1] * v + c;
    sbcMatrix[1][2] = axxis[1] * axxis[2] * v - axxis[0] * s;

    sbcMatrix[2][0] = axxis[0] * axxis[2] * v - axxis[1] * s;
    sbcMatrix[2][1] = axxis[1] * axxis[2] * v + axxis[0] * s;
    sbcMatrix[2][2] = axxis[2] * axxis[2] * v + c;

    c = (float)cos((-angle * M_PI) / 180.0);
    s = (float)sin((-angle * M_PI) / 180.0);
    v = 1.0f - c;

    sbcFlag = 1;

    // [row][col]
    sbcInvMatrix[0][0] = axxis[0] * axxis[0] * v + c;
    sbcInvMatrix[0][1] = axxis[0] * axxis[1] * c - axxis[2] * s;
    sbcInvMatrix[0][2] = axxis[0] * axxis[2] * v + axxis[1] * s;

    sbcInvMatrix[1][0] = axxis[0] * axxis[1] * v + axxis[2] * s;
    sbcInvMatrix[1][1] = axxis[1] * axxis[1] * v + c;
    sbcInvMatrix[1][2] = axxis[1] * axxis[2] * v - axxis[0] * s;

    sbcInvMatrix[2][0] = axxis[0] * axxis[2] * v - axxis[1] * s;
    sbcInvMatrix[2][1] = axxis[1] * axxis[2] * v + axxis[0] * s;
    sbcInvMatrix[2][2] = axxis[2] * axxis[2] * v + c;

    return;
}

void Grid::sbcApply(float *x, float *y, float *z)
{
    float nx, ny, nz;

    nx = sbcMatrix[0][0] * (*x) + sbcMatrix[0][1] * (*y) + sbcMatrix[0][2] * (*z);
    ny = sbcMatrix[1][0] * (*x) + sbcMatrix[1][1] * (*y) + sbcMatrix[1][2] * (*z);
    nz = sbcMatrix[2][0] * (*x) + sbcMatrix[2][1] * (*y) + sbcMatrix[2][2] * (*z);

    *x = nx;
    *y = ny;
    *z = nz;

    return;
}

void Grid::sbcInvApply(float *x, float *y, float *z)
{
    float nx, ny, nz;

    nx = sbcInvMatrix[0][0] * (*x) + sbcInvMatrix[0][1] * (*y) + sbcInvMatrix[0][2] * (*z);
    ny = sbcInvMatrix[1][0] * (*x) + sbcInvMatrix[1][1] * (*y) + sbcInvMatrix[1][2] * (*z);
    nz = sbcInvMatrix[2][0] * (*x) + sbcInvMatrix[2][1] * (*y) + sbcInvMatrix[2][2] * (*z);

    *x = nx;
    *y = ny;
    *z = nz;

    return;
}

void Grid::sbcVel(float *u, float *v, float *w, int sbcIdx)
{
    cerr << "Grid::sbcVel = shouldn't be called" << endl;

    if (sbcIdx > 0)
    {
        while (sbcIdx > 0)
        {
            sbcInvApply(u, v, w); // sieht nicht schlecht aus waehrend dem RK5, allerdings danach siehts komisch aus
            //sbcApply(u,v,w);
            sbcIdx--;
        }
    }
    else if (sbcIdx < 0)
    {
        while (sbcIdx < 0)
        {
            sbcApply(u, v, w);
            //sbcInvApply(u,v,w);
            sbcIdx++;
        }
    }

    return;
}
