filename='results200s';

w=load([filename, '.txt']);
[numOfImages, numOfPixels] = size(w);
inds = randperm(numOfImages);
numOfImages = 64;
d = sqrt(numOfPixels);
rows = floor(sqrt(numOfImages));
cols = ceil(numOfImages / rows);


ha = tight_subplot(rows, cols, [0.01, -0.25], 0.01, 0.01);
for i=1:numOfImages
    axes(ha(i)), imshow(reshape(w(inds(i),:),[d d]),[])
end

print ha tmp.eps -depsc2
