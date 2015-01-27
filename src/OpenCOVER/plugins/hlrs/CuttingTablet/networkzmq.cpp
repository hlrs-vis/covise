/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <errno.h>

#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>

#include "networkzmq.h"
#include "network.pb.h"

Server::Server(int portMatrix, int portMesh)
    : mesh(NULL)
    , num(0)
    , dataSet(0)
{

    char address[64];
    int ret;

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    context = zmq_ctx_new();
    socketMatrix = zmq_socket(context, ZMQ_PULL);
    socketMesh = zmq_socket(context, ZMQ_PUSH);

    snprintf(address, 64, "tcp://*:%d", portMatrix);
    ret = zmq_bind(socketMatrix, address);
    snprintf(address, 64, "tcp://*:%d", portMesh);
    ret = zmq_bind(socketMesh, address);
}

void Server::poll()
{

    zmq_msg_t msg;
    zmq_msg_init(&msg);

    while (zmq_recvmsg(socketMatrix, &msg, ZMQ_DONTWAIT) != -1)
    {

        void *data = ((char *)zmq_msg_data(&msg)) + sizeof(int);
        int size = zmq_msg_size(&msg) - sizeof(int);
        int type = ((int *)zmq_msg_data(&msg))[0];

        switch (type)
        {

        case 1:
        {
            Marker marker;
            marker.ParseFromArray(data, size);

            if (marker.matrix_size() == 16)
            {
                Eigen::Matrix4f matrix;
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        matrix(x, y) = marker.matrix(x + y * 4);

                Eigen::Matrix4f inverse = matrix.inverse();
                Eigen::Matrix4f transpose = inverse.transpose();

                Eigen::Vector3f pos(transpose(0, 3), transpose(1, 3), transpose(2, 3));
                Eigen::Vector3f norm(transpose(0, 2), transpose(1, 2), transpose(2, 2));
                float x = pos(0);
                float y = pos(1);
                float z = pos(2);

                float nx = norm(0);
                float ny = norm(1);
                float nz = norm(2);

                if (marker.mode() == Marker::ENABLE)
                {
                    setPosition(marker.id(), x, y, z);
                    setNormal(marker.id(), nx, ny, nz);
                    /*
		printf("index: %d (%010.4f, %010.4f, %010.4f)\n", marker.id(),
		x, y, z);
	      */
                }
                else if (marker.mode() == Marker::DISABLE)
                {
                    removePosition(marker.id());
                    removeNormal(marker.id());
                }
            }
            break;
        }
        case 2:
        {
            DataSet dataSetMessage;
            dataSetMessage.ParseFromArray(data, size);
            printf("dataSet: %d\n", dataSetMessage.dataset());
            dataSet = dataSetMessage.dataset();
            break;
        }
        }
        zmq_msg_close(&msg);
        zmq_msg_init(&msg);
    }

    if (errno != EAGAIN)
    {

        printf("     [%s]\n", zmq_strerror(errno));
    }

    zmq_msg_close(&msg);
}

void Server::setMarkerMatrix(const int id, const osg::Matrix &matrix)
{

    Marker marker;
    marker.set_id(id);
    marker.set_mode(Marker_MODE_ENABLE);

    for (int index = 0; index < 16; index++)
        marker.add_matrix(matrix.ptr()[index]);

    int size = marker.ByteSize() + sizeof(int);
    zmq_msg_t reply;
    zmq_msg_init_size(&reply, size);
    char *ptr = (char *)zmq_msg_data(&reply);
    ((int *)ptr)[0] = MSG_MARKER;
    marker.SerializeToArray(ptr + sizeof(int), size - sizeof(int));
    int res = zmq_sendmsg(socketMesh, &reply, 0);
    zmq_msg_close(&reply);
}

void Server::setMarkerPosition(const int id,
                               const float px, const float py, const float pz)
{

    Marker marker;
    marker.set_id(id);
    marker.set_mode(Marker_MODE_ENABLE);

    marker.add_matrix(1.0);
    marker.add_matrix(0.0);
    marker.add_matrix(0.0);
    marker.add_matrix(0.0);

    marker.add_matrix(0.0);
    marker.add_matrix(1.0);
    marker.add_matrix(0.0);
    marker.add_matrix(0.0);

    marker.add_matrix(0.0);
    marker.add_matrix(0.0);
    marker.add_matrix(1.0);
    marker.add_matrix(0.0);

    marker.add_matrix(px);
    marker.add_matrix(py);
    marker.add_matrix(pz);
    marker.add_matrix(1.0);

    //markerPosition[id] = osg::Vec3(px, py, pz);

    int size = marker.ByteSize() + sizeof(int);
    zmq_msg_t reply;
    zmq_msg_init_size(&reply, size);
    char *ptr = (char *)zmq_msg_data(&reply);
    ((int *)ptr)[0] = MSG_MARKER;
    marker.SerializeToArray(ptr + sizeof(int), size - sizeof(int));
    zmq_sendmsg(socketMesh, &reply, ZMQ_DONTWAIT);
    zmq_msg_close(&reply);
}

void Server::setNormal(const int id,
                       const float nx, const float ny, const float nz)
{

    normal[id] = osg::Vec3(nx, ny, nz);
}

osg::Vec3 Server::getNormal() const
{

    int num = 0;
    osg::Vec3 n;
    std::map<int, osg::Vec3>::const_iterator i;

    for (i = normal.begin(); i != normal.end(); i++)
    {

        num++;
        n = n + i->second;
    }

    return n / num;
}

void Server::removeNormal(const int id)
{

    normal.erase(id);
}

void Server::setPosition(const int id,
                         const float px, const float py, const float pz)
{

    std::map<int, osg::Vec3>::const_iterator i = markerPosition.find(id);
    /*
  if (i == markerPosition.end()) {

     std::cerr << "Marker " << id << " is not configured" << std::endl;
  } else {
  */
    position[id] = osg::Vec3(px + i->second.x(), py + i->second.y(), pz + i->second.z());
    //}
}

osg::Vec3 Server::getPosition() const
{

    int num = 0;
    osg::Vec3 p;
    std::map<int, osg::Vec3>::const_iterator i;

    for (i = position.begin(); i != position.end(); i++)
    {

        num++;
        p = p + i->second;
    }

    return p / num;
}

void Server::removePosition(const int index)
{

    position.erase(index);
}

void Server::sendGeometry(const int numVertices, const float *vertices,
                          const int numIndices, const unsigned int *indices,
                          const int numTexCoords, const float *texCoords)
{

    /* PROTOBUF
   Mesh m;
   for (int index = 0; index < numVertices * 3; index ++)
      m.add_vertices(vertices[index]);

   for (int index = 0; index < numIndices; index ++)
      m.add_primitives(indices[index]);

   for (int index = 0; index < numVertices; index ++)
      m.add_texcoords(texCoords[index]);

   int size = m.ByteSize();
   zmq_msg_t reply;
   zmq_msg_init_size(&reply, size);
   m.SerializeToArray(zmq_msg_data(&reply), size);
   zmq_sendmsg(socketMesh, &reply, ZMQ_DONTWAIT);
   zmq_msg_close(&reply);
  */

    int size = sizeof(int) + sizeof(int) + sizeof(int) + numVertices * 3 * sizeof(float) + numIndices * sizeof(int) + numVertices * sizeof(float);
    /*
  printf("sending %d bytes (%d %d %d)\n", size,
         numVertices, numIndices, numTexCoords);
  */
    zmq_msg_t reply;
    zmq_msg_init_size(&reply, size);
    char *ptr = (char *)zmq_msg_data(&reply);

    ((int *)ptr)[0] = (int)MSG_MESH;
    ptr += sizeof(int);
    memcpy(ptr, &numVertices, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &numIndices, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, vertices, numVertices * 3 * sizeof(float));
    ptr += numVertices * 3 * sizeof(float);
    memcpy(ptr, indices, numIndices * sizeof(int));
    ptr += numIndices * sizeof(int);
    memcpy(ptr, texCoords, numVertices * sizeof(float));
    zmq_sendmsg(socketMesh, &reply, ZMQ_DONTWAIT);
    zmq_msg_close(&reply);
}

int Server::getDataSet()
{

    return dataSet;
}
