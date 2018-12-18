# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:18:44 2018

@author: Florian
"""
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import warnings
import struct
import os


def loadIES(fname):
    print("loading: "+fname)
    fid = open(fname)
    PhotometricData = {"header": []}

    while True:
        line = fid.readline()
        PhotometricData["header"].extend([line])
        if line.find('TILT') != -1:
            break

    keys = ["# of lamps", "lumens/lamp", "multiplier", "# of vertical angles",
            "# of horizontal angles", "photometric type", "unj-ts type",
            "width", "length", "height", "ballast factor",
            "ballast lamp photometric factor", "input watts"]
    values = []
    for i, key in enumerate(keys):
        # values may be written in several lines...
        if i == len(values):
            values.extend(fid.readline().split(' '))
        value_string = values[i].replace("\n", "")
        if '.' in value_string:
            value = float(value_string)
        else:
            value = int(value_string)
        PhotometricData[key] = value
        print(i, key, value_string)
    arr = np.loadtxt(StringIO(fid.read().replace('\n', ' ')))
    fid.close()

    v_angle = PhotometricData["# of vertical angles"]
    h_angle = PhotometricData["# of horizontal angles"]

    PhotometricData["vertical angles"] = arr[:v_angle]
    PhotometricData["horizontal angles"] = arr[v_angle:(v_angle + h_angle)]
    candela = arr[v_angle + h_angle:]
    PhotometricData["candela values"] = candela.reshape((h_angle, -1))
    elements = PhotometricData["candela values"].shape[1]
    if elements != v_angle:
        msg = "expected {:.0f} rows, but got {:.0f} rows".format(v_angle,
                                                                 elements)
        warnings.warn(msg)
    return PhotometricData


def to_struct(left, bottom, width, height, t_width, t_height, t_depth, data):
    """
    """
    extent = (left, left+width, bottom, bottom+height)
    plt.imshow(np.sum(data, axis=0), interpolation="nearest", cmap="gray",
               origin="lower", extent=extent, vmin=0, vmax=255)
    plt.show()
    data.shape = data.size  # flatten data
    # '?' -> _BOOL , 'h' -> short, 'i' -> int and 'l' -> long
    binary = struct.pack('ffffiii', left, bottom, width, height, t_width,
                         t_height, t_depth) + data.tobytes()
    # t_width.to_bytes(1, 'big')
    print("left:", left)
    print("bottom:", bottom)
    print("width:", width)
    print("height:", height)
    print("texture width:", t_width)
    print("texture height:", t_height)
    print("texture depth:", t_depth)
    print("first element:", data[0])
    print("second element:", data[1])
    print("last element:", data[-1])
    return binary


def plot(ies):
    x = ies["horizontal angles"]
    y = ies["vertical angles"]
    z = ies["candela values"]
    z.shape = (x.shape[0], y.shape[0])
    extent = (x.min(), x.max(), y.min(), y.max())
    plt.imshow(z.T, interpolation="nearest", extent=extent, cmap="gray",
               origin="lower")
    plt.show()


def rescale(data):
    """Transposes a 3D input array to the right order.
    Rescales all values in data to a value range 0 to 255.
    Casts data to unsigned 8 bit integer.
    Returns a contiguous array in memory (C order).
    """
    assert data.ndim == 3, "data should be 3D"
    data = data.transpose(2, 0, 1)  # makes a deep-copy
    min_ = np.min(data)
    max_ = np.max(np.sum(data, axis=0))
    range_ = max_ - min_
    assert (range_ > 0.0), "data set contains only one value."
    rescaled_data = data-min_
    rescaled_data *= (255/range_)
    return np.ascontiguousarray(np.round(rescaled_data, decimals=0), np.uint8)


def ies2mlb(path, n_lights):
    names_pl = [x for x in os.listdir(path) if x.endswith(".ies")]
    assert n_lights <= len(names_pl), "requested more photometric lights than available in the path "+path

    test_ies = loadIES(path+names_pl[0])
    t_height = test_ies["# of vertical angles"]
    t_width = test_ies["# of horizontal angles"]
    an_hor = test_ies["horizontal angles"]
    an_ver = test_ies["vertical angles"]

    max_texture_size = 2048
    if t_width > max_texture_size:
        cut = (t_width-max_texture_size)//2+1
        hs, he = cut, (t_width-cut)
        an_hor = an_hor[hs:he]
        t_width = len(an_hor)
        warnings.warn("texture will be cropped! {:.0f} pixels from the left and right are cut off.".format(cut))
    else:
        hs, he = 0, t_width
    if t_height > max_texture_size:
        cut = (t_height-max_texture_size)//2+1
        vs, ve = cut, (t_height-cut)
        an_ver = an_ver[vs: ve]
        t_height = len(an_ver)
        warnings.warn("texture will be cropped! {:.0f} upper and lower pixels are cut off.".format(cut))
    else:
        vs, ve = 0, t_height

    left = np.radians(an_hor[0])
    bottom = np.radians(an_ver[0])
    width = np.radians(an_hor[-1]-an_hor[0])
    height = np.radians(an_ver[-1] - an_ver[0])

    all_pl = np.zeros((t_height, t_width, n_lights))
    for i in range(n_lights):
        photometric_light = loadIES(path+names_pl[i])
        all_pl[:, :, i] = (photometric_light["candela values"].T)[vs:ve, hs:he]

    data_byte = rescale(all_pl)
    return to_struct(left, bottom, width, height, t_width, t_height, n_lights,
                     data_byte)


def img2mlb(p):
    img = plt.imread(p)
    t_height, t_width, t_depth = img.shape
    img = np.ascontiguousarray(img[::-1, :, :])
    data_byte = rescale(img)
    w, h = np.radians(30), np.radians(20)
    return to_struct(-w/2, -h/2, w, h, t_width, t_height, t_depth, data_byte)


def example_ies2mlb():
    p = "D:/covise_stuff/MatrixScheinwerfer/96P_Lichtverteilung+Vorfeld/"
    binary = ies2mlb(p, 96)
    fh = open("D:/covise_stuff/MatrixScheinwerfer/PhotometricMatrixLights.mlb",
              "wb")
    fh.write(binary)
    fh.close()
    return


def example_img2mlb():
    binary = img2mlb("D:/covise_stuff/sample_small.png")
    fh = open("D:/covise_stuff/MatrixScheinwerfer/test.mlb", "wb")
    fh.write(binary)
    fh.close()
    return


if __name__ == "__main__":
    example_img2mlb()
#    example_ies2mlb()
    print("done.")
