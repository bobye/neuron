w=load('results25.txt');
[numOfImages, numOfPixels] = size(w);
inds = randperm(numOfImages);
numOfImages = 25;
d = sqrt(numOfPixels);
rows = floor(sqrt(numOfImages));
cols = ceil(numOfImages / rows);

for i=1:numOfImages
    subplot(rows, cols, i), imshow(reshape(w(inds(i),:),[d d]),[])
end
