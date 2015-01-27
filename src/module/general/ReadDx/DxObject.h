/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DXOBJECT_DEFINED
#define __DXOBJECT_DEFINED

/** Container class describing an object in IBM
 * Data Explorer
 */
class DxObject
{
private:
    char *name_;
    int objectClass_;
    int dataOffset_;
    int rank_;
    int items_;
    int type_;
    int shape_;
    int dataFormat_;
    int byteOrder_;
    char *fileName_;
    char *elementType_;
    char *ref_;
    char *data_;
    char *connections_;
    char *positions_;
    char *attributeDep_;
    char *attributeName_;
    bool follows_;
    void setMember(char *&member, const char *value);
    void setMember(char *&member, int number);

public:
    /// Constructor
    /** @param defaultByteOrder describes the byte order of the data to be read in
       * ( and not the machine the code is running on ).
       * This value is used when the byte order is not specified in the dx File
       */
    DxObject(int defaultByteOrder);
    /**
       * @param fileName set the name of the file containing the data
                of the dx object
       */
    void setFileName(const char *fileName)
    {
        setMember(this->fileName_, fileName);
    }

    /** @param elementType set the element type of the dx object if it is of the object class
       *  gridpositions.
       * Currently accepted are: "cubes", "quads", "triangles"
       */
    void setElementType(const char *elementType)
    {
        setMember(this->elementType_, elementType);
    }
    /** @param ref set the reference attribute of the dx object
       */
    void setRef(const char *ref)
    {
        setMember(this->ref_, ref);
    }

    /** @param data determine the name of the data component of a dx object which is a field
       */

    void setData(const char *data)
    {
        setMember(this->data_, data);
    }
    /** @param data determine the name of the data component of a dx object which is a field or multigrid
       * the numeric value of data is transformed to a unique string.
       */
    void setData(int data)
    {
        setMember(this->data_, data);
    }

    /** @param connections determine the name of the connections component of a dx object which is a field or multigrid
       */
    void setConnections(const char *connections)
    {
        setMember(this->connections_, connections);
    }
    /** @param connections determine the name of the connections component of a dx object which is a field or multigrid
       * the numeric value of connections is transformed to a unique string.
       */
    void setConnections(int connections)
    {
        setMember(this->connections_, connections);
    }

    /** @param positions determine the name of the positions component of a dx object which is a field or multigrid
       */
    void setPositions(const char *positions)
    {
        setMember(this->positions_, positions);
    }
    /** @param positions determine the name of the positions component of a dx object which is a field or multigrid
       * the numeric value of positions is transformed to a unique string.
       */
    void setPositions(int positions)
    {
        setMember(this->positions_, positions);
    }

    /** @param attributeDep set the "dep" attribute of a dx object
       */
    void addAttributeDep(const char *attributeDep)
    {
        setMember(this->attributeDep_, attributeDep);
    }

    /** @param attributeName set the name attribute of a dx object
       */
    void addAttributeName(const char *attributeName)
    {
        setMember(this->attributeName_, attributeName);
    }

    /** @param name determine the name of a dx object.
       */
    void setName(const char *name)
    {
        setMember(this->name_, name);
    }
    /** @param name determine the name of a dx object.
       * the numeric value of name is transformed to a unique string.
       */
    void setName(int name)
    {
        setMember(this->name_, name);
    }

    /** @param follows  true if the data belonging to a dx object follow it's specification
       * in the dx file
       * note that reading dx files build like that is not yet implemented
       */
    void setFollows(bool follows)
    {
        this->follows_ = follows;
    }

    /** @param rank determine the rank of a dx object
       * rank 0 corresponds to scalar values, the shape is then always 1
       * rank 1 corresponds to vectors. Examples:
       * An unstructured grid consisting
       * of 17 cubes has  rank 1 shape 8 items 17
       * rank 2 which is not implemented corrsponds to matrices or tensors
       * a scalar data object belonging to this grid would have
       * "rank 0 items 17"
       */
    void setRank(int rank)
    {
        this->rank_ = rank;
    }
    /** @param items the number of elements a dx object consists of
       */
    void setItems(int items)
    {
        this->items_ = items;
    }

    /** @param type determine the type of data a dx object consists of
       * possible values are Parser::DOUBLE Parser::FLOAT Parser::INT  Parser::UINT
       * Parser::SHORT Parser::USHORT Parser::BYTE Parser::UBYTE
       */
    void setType(int type)
    {
        this->type_ = type;
    }
    /** @param shape set the shape of a dx object
       */
    void setShape(int shape)
    {
        this->shape_ = shape;
    }

    /** @param format determine the format of the data which can be
          Parser::ASCII or Parser::BINARY
       */
    void setDataFormat(int format)
    {
        this->dataFormat_ = format;
    }

    /** @param order determine the byte order of the data to be read in
       * possible values are Parser::LSB for little endian data
       * and Parser::MSB for big endian data
       * this parameter is of interset only if the data format is Parser::BINARY
       */
    void setByteOrder(int order)
    {
        this->byteOrder_ = order;
    }

    /** @param objectclass determine the class of the object which may be
       * Parser::FIELD, Parser::ARRAY, Parser::MULTIGRID and Parser::GROUP
       *
       */
    void setObjectClass(int objectclass)
    {
        this->objectClass_ = objectclass;
    }

    /** @param offset determine the offset in the data file where to find
       * the data corresponding to this dx object
       */
    void setDataOffset(int offset)
    {
        this->dataOffset_ = offset;
    }

    /** Get the 'follows' attribute of the object
       * @return true if "follows" was set
       */
    bool getFollows()
    {
        return this->follows_;
    }

    /** Get the rank of a data object
       * @return the rank
       */
    int getRank()
    {
        return this->rank_;
    }
    /** Get the number of items a dx object consists of
       * @return number of items
       */
    int getItems()
    {
        return this->items_;
    }

    /** Traditional 'Get' method
       * @return either Parser::DOUBLE, Parser::FLOAT, Parser::INT,  Parser::UINT,
       * Parser::SHORT, Parser::USHORT, Parser::BYTE or Parser::UBYTE
       */
    int getType()
    {
        return this->type_;
    }

    /** Traditional 'Get' method
       * @return shape of the dx object
       */
    int getShape()
    {
        return this->shape_;
    }

    /** Traditional 'Get' method
       * @return Byteorder of the data corrsponding to the dx object
       * either Parser::MSB or Parser::LSB
       */
    int getByteOrder()
    {
        return this->byteOrder_;
    }

    /** Traditional 'Get' method
       * @return offset in file where data start
       */
    int getDataOffset()
    {
        return this->dataOffset_;
    }

    /** Traditional 'Get' method
       * @return either    Parser::FIELD, Parser::ARRAY, Parser::MULTIGRID or Parser::GROUP
       */
    int getObjectClass()
    {
        return this->objectClass_;
    }

    /** Traditional 'Get' method
       * @return Parser::ASCII or Parser::BINARY
       */
    int getDataFormat()
    {
        return this->dataFormat_;
    }

    /** Traditional 'Get' method
       * @return the reference attribute of the object
       */
    const char *getRef()
    {
        return this->ref_;
    }

    /** Traditional 'Get' method
       * @return the name attribute of the dx-object
       */
    const char *getAttributeName()
    {
        return this->attributeName_;
    }

    /** Traditional 'Get' method
       * @return the 'dep' attribute of the dx-object
       */
    const char *getAttributeDep()
    {
        return this->attributeDep_;
    }

    /** Traditional 'Get' method
       * @return the name of the dx object
       */
    const char *getName()
    {
        return this->name_;
    }

    /** Traditional 'Get' method
       * @return name of the data component if this object is a field
       */
    const char *getData()
    {
        return this->data_;
    }

    /** Traditional 'Get' method
       * @return the name of the connections component of a dx object if it is a field
       */
    const char *getConnections()
    {
        return this->connections_;
    }

    /** Traditional 'Get' method
       * @return  the name of the positions component of a dx object if it is a field
       */
    const char *getPositions()
    {
        return this->positions_;
    }

    /** Traditional 'Get' method
       * @return the type of the elements, currently either "cubes", "quads" or triangles
       */
    const char *getElementType()
    {
        return this->elementType_;
    }

    /** Traditional 'Get' method
       * @return the filename of the file containing the data
       * of this dx object
       */
    const char *getFileName()
    {
        return this->fileName_;
    }

    /** Show all members of the dx object, for testing purposes only
       */
    void show();
};

typedef enum
{
    quads,
    cubes,
    triangles,
    tetrahedra
} ElementType;
#endif
