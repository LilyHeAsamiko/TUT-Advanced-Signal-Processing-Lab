% Reads the light intensity value from DrDAQ board. Values will be in the
% scale [0,100].
% NOTE: The board must be connected first before calling this function,
% i.e. run script initMeasCard.m first.
function intensityvalue = getIntensity(picohandle)

% Channel number 7 corresponds to light
lightchannel = 7;
outval = libpointer('int16Ptr', -1);
overflowb = libpointer('uint16Ptr', -1);
[picostatus,intensityvalue,~] = calllib('USBDrDAQ','UsbDrDaqGetSingle',...
    picohandle,lightchannel,outval,overflowb);

if picostatus
    disp('Error in reading intensity value. Check handle.');
    intensityvalue = -1;
    return
end
% Divide intensity value to make it have the correct scale
intensityvalue = double(intensityvalue);
intensityvalue = intensityvalue / 10.0;
