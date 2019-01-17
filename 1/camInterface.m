% SGN-26006 - Light intensity exercise
%
% This script gives some commands that are needed to control the
% webcam. More information about controlling imaging 
% devices can be found in Matlab help under Image Acquisition Toolbox

% Gives a list of available adaptors
imaqhwinfo
% We'll use 'winvideo'
% more info about that is available using the following command:
imadapt = imaqhwinfo('winvideo');

% Select the adaptor and some of the image formats (use RGB color space):
% vid = videoinput('winvideo',1,'RGB24_800x600');
% All available possibilities: imadapt.DeviceInfo.SupportedFormats)
vid = videoinput('winvideo',1,'MJPG_800x600');

% You may check some properties of the video input object
get(vid)

source = getselectedsource(vid);
% Look at the parameters of the active video source
get(source)

% Set focus manually
%set(source,'Focus','manual');
set(source,'Focus',25);

% Get more info about a specific parameter (e.g. its range of values)
propinfo(source,'Exposure')

% Set at least these to allow manual control of the camera exposure
set(source,'ExposureMode','manual')
source.Exposure=-9;
set(source,'BacklightCompensation','off')
set(source,'WhiteBalanceMode','manual')

% To see preview of the video stream, you can open a preview
%preview(vid)

% To close the preview:
%closepreview(vid)

% Before you can get images, the video source must be started
start(vid)

% Capturing one image
im = getdata(vid,1);
imwrite(im, ['img_expo_' num2str(source.Exposure) 'int_' num2str(getIntensity(ph)) '.jpeg'])