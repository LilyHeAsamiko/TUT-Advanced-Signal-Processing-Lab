function [y2_,lookup]=histequal(I,k)
[m,n]=size(I);
hist_=zeros(1,k);
for i=1:m
    for j=1:n
    hist_(I(i,j)+1)=hist_(I(i,j)+1)+1;
    end
end

p(1)=0;
for i=1:k
    p(i+1)=p(i)+hist_(i)/sum(hist_(:));
    lookup(i)=round(255*p(i+1)+0.5);
end

for i=1:m
    for j=1:n
    y2_(i,j)=lookup(I(i,j)+1);
    end
end

end