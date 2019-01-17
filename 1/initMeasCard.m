% Initializes the connection to the DrDAQ board  
function picohandle = initMeasCard
addpath('D:\SGN_26006_lightintensity_assignment\USBDrDAQ_SDK');
loadlibrary('USBDrDAQ.dll', ...
    'usbDrDaqApi.h');
libfunctions('USBDrDAQ','-full')
 
[picostatus, picohandle] = calllib('USBDrDAQ','UsbDrDaqOpenUnit', 1);

if picostatus
    disp('Problem in initializing measurement board.');
else
    disp('Measurement board OK.');
end

end
