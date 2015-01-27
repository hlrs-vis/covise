# This is a dummy file required by *Base.py.
# Don't add anything and don't remove the file!

# Background:
# The PyQt ui converter expects the resources to be in StaticImages_rc.py
# (because the resource file is calles StaticImages.qrc).
# However, we want to load the resources dynamically based on the configuration.
# Therefore we give the *Base.py files an empty StaticImages_rc.py and load the
# binary *.rcc files manually.

