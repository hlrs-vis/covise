try:
    import cPickle as pickle
except:
    import pickle

pickleFile = coviseDatenDir + '/' + cocasename + '.cocase'
output = open(pickleFile, 'wb')
pickle.dump(cocase,output)
output.close()

logFile.write("\ncocasefile = %s\n"%(pickleFile,))
logFile.flush()
print("cocase file = %s"%(pickleFile,))

CoviseMsgLoop().unregister(aErrorLogAction)

logFile.write("\nConversion finished\n")
logFile.flush()
logFile.close()
print("\nConversion finished see log file %s\n"%(logFileName,))

sys.exit()
