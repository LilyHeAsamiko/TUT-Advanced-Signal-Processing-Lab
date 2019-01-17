%% lf_compress Hints

    %Encoding Intensity   
    % Draw the histogram of the values found in the matrix A
    [R,C]=size(A);
    %Define the range of the bins values 
    maxA = max(A(:)); %0
    minA = min(A(:)); %215
    %Draw the histogram (hint:histc)
    histA = histc(A(:),0:215);
    figure,
    plot(histA);
    % Calculating entropy
    [counts,binLocations] = imhist(A);
    p = counts/sum(counts);
    if p == 0
        Entropy = 0;
    else 
        Entropy = -sum(nonzeros(p).*log(nonzeros(p)));
    end
    %Remember that the entropy calculation is based on  
    %the number of gray levels and the probability pk associated with each
    %gray level k (hint: imhist, histogram)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%ENCODING FUNCTION CALL GOES HERE

%HINTS:

% Golomb-Rice Coding Hints

A = double(A);

% Compute the residual matrix E
[rows,columns] = size(A);
E = zeros(size(A));

%since in A there isn't any north, west and north west values it's better to put first
%value of E same as be A

E = A;

%Values which are in first row of A matrix do not have north and northwest
%values,so the west value is put likewise for values in first column


Z_hat = zeros(size(A));

%Analysis for first row and column

for c = 2:columns
    E(1,c) = median([0,E(1,c-1),E(1,c-1),E(1,c-1),E(1,c-1)]);
end

for r = 2:rows
    E(r,1) = median([E(r-1,1),0,E(r-1,1),E(r-1,r)*((E(r,r+1)-1)),E(r,r+1)*E(r-1,r)/2]);   
end

%Finding other E values from second row and second column
for i = 2:rows
    for j = 2:columns
        E(i,j) = median([E(i-1,j),E(i,j-1),E(i-1,j)+E(i,j-1)-E(i-1,j)*E(i,j-1),E(i,j-1)-E(i-1,j)+E(i-1,j)*E(i,j-1),E(r,r+1)*E(r-1,r)/2]);
    end
end

figure,
plot(E)

%Divide the matrix E into blocks of size b × b and collect in a vector e

%Hints:

H = size(A,1);
W = size(A,2);
NumberOfblcksInHeight = H/b;
NumberOfblcksInWidth = W/b;


%Golomb-Rice step

%Read carefully step 5 of the pdf to find the key parameters to solve the
%GR coding


%Step 1: find the optimal p for each block (remember that optimum parameter p is close to log2 of the mean of the absolute value in the block)
for h = 1 : b : H    
    for w = 1 : b: W
        for t = 0 : 8
            p=min(min(1+t+A(w:w+b,h:h+b)/2^t+1));
            A(w:w+b,h:h+b)= p;
        end
    end
end 
figure,
imshow(A), colormap(gray)
%Step 2: use the p to compute the binary translation of the value



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Encoding depth
    
%     Depth = im2uint8(input_matrix);
    Depth = A;
    % Quantizing the depth values
    minD = min(min(Depth));
    maxD = max(max(Depth));
    D = uint8( floor (255* (Depth - minD)/(maxD - minD)) );
    
%Try to apply the steps used for the intensity image to the Depth Image and
%encode it with the new quantization

% REMEMBER THAT IN THE ENCODED DATA STRUCTURE YOU NEED ALL THE CORE
% INFORMATIONS (P VALUE, BLOCK SIZE, etc...)
% TO RECONSTRUCT THE DATA AT THE DECODER


%% lf_decompress Hints


% The input of this function in encoded bitsream and return the decoded
% signal as an output.
%
%Hint: try to follow the steps in the coding function in reverse order to
%obtain the desired output for the Depth Matrix
%
%