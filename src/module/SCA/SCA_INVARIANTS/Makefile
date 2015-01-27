# Makefile for copying SCA invariant files
DIRECTORIES = LSDYNA MATERIALS VISUALISATION TRACTION

all:
	cp -r $(DIRECTORIES) $(SCA_PATH)
	chmod -R u+xw $(SCA_PATH)

clean:
	/bin/rm -fr $(SCA_PATH)/LSDYNA
	/bin/rm -fr $(SCA_PATH)/MATERIALS
	/bin/rm -fr $(SCA_PATH)/VISUALISATION
	/bin/rm -fr $(SCA_PATH)/TRACTION

