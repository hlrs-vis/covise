from pymol import cmd
from pymol import preset
from pymol import util
from glob import glob
import sys
import os

path = ""
params = []

basepath = os.getcwd()

try:
    path = sys.argv[len(sys.argv) - 1]
    print path
    cmd.cd(path)
    #index = sys.argv.index("--")
    #params = sys.argv[index:]
    #if(len(params) == 2):
        #path = params[1]
        #cmd.cd(path)
    #else:
        #print "No Path specified"
except ValueError:
    print "No Path specified"

for file in glob("*.pdb"):
    print "file: ", file
    listname = file.split(".")
    name = listname[0];
    cmd.load(file, name)
    cmd.system("mv " + file + " ../pdb/")
    cmd.hide("all")
    cmd.show("sticks")
    cmd.reset()
    cmd.origin(position=[0.0,0.0,0.0])
    cmd.save(name + "stix.wrl")
    cmd.hide("all")
    cmd.show("ribbon")
    cmd.reset()
    cmd.origin(position=[0.0,0.0,0.0])
    cmd.save(name + "rib.wrl")
    cmd.hide("all")
    preset.pretty(name)
    cmd.reset()
    cmd.origin(position=[0.0,0.0,0.0])
    cmd.save(name + "cart.wrl")
    cmd.hide("all")
    cmd.show("surface")
    cmd.reset()
    cmd.origin(position=[0.0,0.0,0.0])
    cmd.save(name + "surf.wrl")
    cmd.delete("all")
    #cmd.system("rm -f " + file)
    #cmd.system("mv " + file + " ../pdb/")
    cmd.system("wget -nc http://www.pdb.org/pdb/images/" + name.lower() + "_bio_r_250.jpg")
    cmd.system("convert " + name.lower() + "_bio_r_250.jpg " + name + ".tif")
    cmd.system("rm -f " + name.lower() + "_bio_r_250.jpg")
    cmd.system("mv " + name + ".tif" + " ../pdb/")
    print "Created " + name + " models"
cmd.cd(basepath)
