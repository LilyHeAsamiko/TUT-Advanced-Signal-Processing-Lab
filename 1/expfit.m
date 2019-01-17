function ExposureInd = expfit(Intensity)
    ExposureInd = -8.202*exp(0.001991*Intensity)+9.948*exp(-0.1366*Intensity);
    for i = 1:length(Intensity)
        if ExposureInd(i) >-2
            ExposureInd(i) = -2;
        elseif ExposureInd(i) < -11
            ExposureInd(i) = -11;
        end
    end
    display(ExposureInd);
    figure
    plot(ExposureInd);
end