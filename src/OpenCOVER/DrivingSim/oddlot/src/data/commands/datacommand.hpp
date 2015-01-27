/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DATACOMMAND_HPP
#define DATACOMMAND_HPP

#include <QUndoCommand>

class DataCommand : public QUndoCommand
{
public:
    explicit DataCommand(QUndoCommand *parent = NULL);
    virtual ~DataCommand()
    {
    }

    virtual int id() const = 0;

    virtual void undo() = 0;
    virtual void redo() = 0;

    bool isValid() const
    {
        return isValid_;
    }
    bool isUndone() const
    {
        return isUndone_;
    }

protected:
    void setInvalid()
    {
        isValid_ = false;
    }
    void setValid()
    {
        isValid_ = true;
    }

    void setUndone()
    {
        isUndone_ = true;
    }
    void setRedone()
    {
        isUndone_ = false;
    }

private:
    //	DataCommand(); /* not allowed */
    DataCommand(const DataCommand &); /* not allowed */
    DataCommand &operator=(const DataCommand &); /* not allowed */

private:
    bool isUndone_;
    bool isValid_;
};

#endif // DATACOMMAND_HPP
