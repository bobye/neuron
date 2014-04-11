%% data load
A = load('mturkData.txt');
B = load('mturkData-Decode.txt');

n = size(A,1);

%% scatter plot
S = load('mturkData-s.txt');
scatplot(S, sum((A-B)'.^2),'circles')
%% sample view
i = randi(n,1);

cpo = reshape(reshape(A(i,:),3,5)', 1, 5, 3);
cpd = reshape(reshape(B(i,:),3,5)', 1, 5, 3);

figure
subplot(2,1,1); imshow(cpo);
subplot(2,1,2); imshow(cpd);