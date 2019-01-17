  function [m, v] = slidewindow(Img, h, w)
  [r,c] = size(Img);
  for i = 1:c-h
      for j = 1:r-w
          M = Img(i:i+h-1,j:j+w-1);
          m(i) = mean(M(:));
          v(i) = sum(double(M(:))-repmat((m(i)),h*w,1))/(h*w);
      end
  end
  figure,
  scatter(m,v);
  end