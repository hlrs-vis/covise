if ( ! $?COVISEDIR ) then
        echo "COVISEDIR not set (Did you source .covise?)"
	exit 1
endif

set DBSH=/tmp/${USER}-dot-env.sh
set DCSH=/tmp/${USER}-dot-env.csh
set DCSH2=/tmp/${USER}-dot-env-pre.csh
set DCSH3=/tmp/${USER}-dot-env-post.csh

# Create helper script
cat ${COVISEDIR}/scripts/covise-env.sh > "$DBSH"
echo "printenv | cut -b 1-4099" >> "$DBSH"

# Make a snapshot of the previous environment
printenv | sed -e 's/^/setenv /' -e 's/=/ "/' -e 's/$/"/' | sort > "$DCSH2"

# Execute the helper script to get the COVISE environment 
bash "$DBSH" | sed -e 's/^/setenv /' -e 's/=/ "/' -e 's/$/"/' | sort > "$DCSH3"

# Compare both results to extract the differences
diff "$DCSH3" "$DCSH2" | grep '^<' | cut -f2- -d\  > "$DCSH"

rm -f "$DBSH" "$DCSH2" "$DCSH3"

source "$DCSH"
rm -f "$DCSH"

unsetenv GLOBUS_DEFINES
#unsetenv GLOBUS_LIBS
