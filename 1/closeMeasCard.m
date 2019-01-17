% Closes the connection to the DrDAQ board
function closeMeasCard(picohandle)
picostatus = calllib('USBDrDAQ','UsbDrDaqCloseUnit',picohandle);
if picostatus
    disp('Could not close measurement board. Check handle.');
else
    disp('Measurement board closed.');
end
