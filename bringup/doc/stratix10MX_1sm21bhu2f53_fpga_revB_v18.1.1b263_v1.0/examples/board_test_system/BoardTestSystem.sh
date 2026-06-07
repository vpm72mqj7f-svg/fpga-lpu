#!/bin/sh
if [ "$QUARTUS_ROOTDIR" == "" ]; then
	export QUARTUS_ROOTDIR=/tools/acds/15.1.2/current.linux/linux64/quartus
	echo "Use default QUARTUS_ROOTDIR: $QUARTUS_ROOTDIR"
fi
export PATH=$QUARTUS_ROOTDIR/linux64/jre64/bin:$PATH
java -Xmx256m -jar bts.jar 
