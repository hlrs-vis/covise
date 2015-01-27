#
# Translation functions for vr-prepare
# Visenso GmbH
# (c) 2012
#
# $Id: TestPathTranslator.py 704 2012-11-20 10:27:29Z wlukutin $

import PathTranslator
import os
import shutil
import unittest

class TestPathTranslator(unittest.TestCase):
    
    testdir1 = "test_dir"
    testdir = "test_dir" + os.sep + "fr_FR"
    filename = "file.dat"
        
    def setUp(self):
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        f = open(self.testdir + os.sep + self.filename, "w+")
        f.write("hi")
        f.close()
    
    def tearDown(self): 
        shutil.rmtree(self.testdir)
    
    def test_strip_empty_locale(self):
        l = ""
        self.assertEqual(PathTranslator.strip_locale(l), "")
        
    def test_strip_arbitrary_locale(self):
        l = "de_DE.UTF-8"
        self.assertEqual(PathTranslator.strip_locale(l), "de_DE")
        
    def test_strip_advanced_locale(self):
        l = "en_EN.UTF-8@calendar=test"
        self.assertEqual(PathTranslator.strip_locale(l), "en_EN")
        
    def test_strip_advanced_2_locale(self):
        l = "en_EN@calendar=test"
        self.assertEqual(PathTranslator.strip_locale(l), "en_EN")

    def test_strip_advanced_path_locale(self):
        l = "..\\locale\\en_EN.UTF-8@calendar=test"
        self.assertEqual(PathTranslator.strip_locale(l), ".." + os.sep + "locale" + os.sep + "en_EN")

    def test_strip_advanced_path_2_locale(self):
        l = "..\\en_EN.UTF-8@calendar=test"
        self.assertEqual(PathTranslator.strip_locale(l), ".." + os.sep + "en_EN")
        
    def test_strip_advanced_path_3_locale(self):
        l = "..\\locale"
        self.assertEqual(PathTranslator.strip_locale(l), ".." + os.sep + "locale")      
        
    def test_translate_path_with_empty_locale(self):
        l = ""
        p = "./" + self.testdir1 + "/non_existing_file.txt"     
        self.assertEqual(PathTranslator.translate_path(l,p), "." + os.sep + "test_dir" + os.sep + "non_existing_file.txt")   
        
    def test_translate_path_with_locale(self):
        l = "fr_FR.UTF-8"
        p = "./" + self.testdir + "/" + self.filename   
        self.assertEqual(PathTranslator.translate_path(l,p), "." + os.sep + "test_dir" + os.sep + "fr_FR" + os.sep + "file.dat")      
        
    def test_dont_translate_path_when_not_existing(self):
        l = "fr_FR.UTF-8"
        p = "./" + self.testdir + "/non_existing_file.txt"   
        self.assertEqual(PathTranslator.translate_path(l,p), "./" + self.testdir + "/non_existing_file.txt")   
        
    def test_dont_alter_path_when_not_existing(self):
        l = "nz_NZ.UTF-8"
        p = "./" + self.testdir + "\\" + self.filename   
        self.assertEqual(PathTranslator.translate_path(l,p), "./" + self.testdir + "\\" + self.filename)      
        
    def test_butufy_path_when_existing(self):
        l = "fr_FR.UTF-8"
        p = "./" + self.testdir + "\\" + self.filename   
        self.assertEqual(PathTranslator.translate_path(l,p), "." + os.sep + self.testdir + os.sep + "fr_FR" + os.sep + self.filename) 

if __name__ == '__main__':
    unittest.main()
