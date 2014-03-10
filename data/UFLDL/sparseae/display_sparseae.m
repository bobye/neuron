w=load('results25.txt');
[numOfImages, numOfPixels] = size(w);
d = sqrt(numOfPixels);
rows = floor(sqrt(numOfImages));
cols = ceil(numOfImages / rows);

for i=1:numOfImages
    subplot(rows, cols, i), imshow(reshape(w(i,:),[d d]),[])
end