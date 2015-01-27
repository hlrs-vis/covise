#!/bin/tcsh

echo "----------------------"


# SERVER_PATH is an optional environment variable which      #
# sets the directory where web_srv search for web files      #   
# - by default is the directory where the web_srv is started #
##############################################################
setenv SERVER_PATH $COVISE_PATH/bin
echo "SERVER_PATH =" $SERVER_PATH


# host where web_srv is started #
#################################
if ( ! $?HOSTSRV) then
    setenv HOSTSRV $COVISE_HOST
endif
echo "HOSTSRV =" $HOSTSRV


# port where web_srv is listening for http connections #
########################################################  
setenv HTTP_PORT 50001
echo "HTTP_PORT =" $HTTP_PORT


# port where web_srv is listening for covise connections #
##########################################################  
setenv COVISE_PORT 51001
echo "COVISE_PORT =" $COVISE_PORT


echo "----------------------"
