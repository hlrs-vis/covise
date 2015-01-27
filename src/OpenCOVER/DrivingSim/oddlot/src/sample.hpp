/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   18.03.2010
**
**************************************************************************/

#ifndef SAMPLE_HPP
#define SAMPLE_HPP

class Sample
{
    //Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    //	explicit Sample();
    virtual ~Sample();

protected:
private:
    Sample(); /* not allowed */
    Sample(const Sample &); /* not allowed */
    Sample &operator=(const Sample &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

    //public slots:

    //public signals:

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // SAMPLE_HPP
