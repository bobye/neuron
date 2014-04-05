A = load('mturkData.txt');
B = load('mturkData-Decode.txt');

n = size(A,1);

i = randi(n,1);

cpo = reshape(reshape(A(i,:),3,5)', 1, 5, 3);
cpd = reshape(reshape(B(i,:),3,5)', 1, 5, 3);

subplot(2,1,1); imshow(cpo);
subplot(2,1,2); imshow(cpd);