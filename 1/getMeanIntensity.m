function sWE = getMeanIntensity( ph )
%GETMEANINTENSITY Summary of this function goes here
%   Detailed explanation goes here
t = 1:100;
s = double(zeros(length(t),1));

for i=t
    s(i) = getIntensity(ph);
end

sWE = mean(s);

end

