% 0 is black and 1 is white!

clearvars; clc; close all;

% Read and Display an Image
originalImage = imread("ImaginiEcografice/imagineTest5.bmp"); % citim imaginea ecografica originala
figure(1), imshow(originalImage), title('Original image to analyze'); % o afisam intr-o figura
hold off

% Extract a bounding box with desired shape to analyze
extractedImage = bwareafilt(originalImage > 0, 1, [1 1 1; 0 1 0; 1 1 1]);
propsExtractedImage = regionprops(extractedImage, 'BoundingBox');
boundingBox = propsExtractedImage.BoundingBox; % punem intr-un Bounding Box ce am extras de pe imaginea ecografica

% Crop original image
croppedImage = imcrop(originalImage, boundingBox); % cropam bounding box-ul din imaginea ecografica originala
figure(2), imshow(croppedImage), title("Cropped Image from original one"); % afisam rezultatul din bounding box

% K means algorithm
numberColors = 4;
[clustedImageWithKmeans, clusterCentroids] = ...
    imsegkmeans(croppedImage, numberColors,'MaxIterations', 100); % aplicam K-Means algorithm 
clusterCentroids = im2double(clusterCentroids); % centroidele fiecarui cluster with K-means (avem 4 in total)

appliedClustedImage = labeloverlay(croppedImage, clustedImageWithKmeans); % adaugam clusterele peste imaginea cropata
figure(3), imshow(appliedClustedImage),
title(['Clustered image with ', num2str(numberColors), ' numColors and ' , ...
    num2str(100), ' iterations']); % afisam imaginea clusterizata

clusteredFinalImage = rgb2gray(appliedClustedImage); % transformam imaginea 3d in 2d pentru a o utiliza mai departe
figure(4), imshow(clusteredFinalImage),
title('Final Clustered Image converted in 2D');



%%