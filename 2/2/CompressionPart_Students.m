%% lf_compress Hints

    %Encoding Intensity   
    % Draw the histogram of the values found in the matrix A
    [R,C]=size(A);
    ________________ %Define the range of the bins values
    ________________ %Draw the histogram (hint:histc)
    
    % Calculating entropy

    %Remember that the entropy calculation is based on  
    %the number of gray levels and the probability pk associated with each
    %gray level k (hint: imhist, histogram)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%ENCODING FUNCTION CALL GOES HERE

%HINTS:

% Golomb-Rice Coding Hints

input_matrix = double(input_matrix);

% Compute the residual matrix E
[rows,columns] = size(input_matrix);
E = zeros(size(input_matrix));

%since in A there isn't any north, west and north west values it's better to put first
%value of E same as be A

_______________________

%Values which are in first row of A matrix do not have north and northwest
%values,so the west value is put likewise for values in first column

___________

Z_hat = zeros(size(input_matrix));

%Analysis for first row and column

for c = 2:columns
    _______________________
end

for r = 2:rows
    ______________________
end

%Finding other E values from second row and second column
for i = 2:rows
    for j = 2:columns
        
    
        __________________
    
    
    end
end


%Divide the matrix E into blocks of size b × b and collect in a vector e

%Hints:
NumberOfblcksInHeight = ceil(size(input_matrix,1)/block_size);
NumberOfblcksInWidth = ceil(size(input_matrix,2)/block_size);
H = size(input_matrix,1);
W = size(input_matrix,2);

for h = 1 : NumberOfblcksInHeight

    
    for w = 1 : NumberOfblcksInWidth
       
    
    end
end

%Golomb-Rice step

%Read carefully step 5 of the pdf to find the key parameters to solve the
%GR coding


%Step 1: find the optimal p for each block (remember that optimum parameter p is close to log2 of the mean of the absolute value in the block)

%Step 2: use the p to compute the binary translation of the value



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Encoding depth
    
%     Depth = im2uint8(input_matrix);
    Depth = input_matrix;
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


