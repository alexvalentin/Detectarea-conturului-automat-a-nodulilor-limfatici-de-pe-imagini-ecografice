% 0 is black and 1 is white!

clearvars; clc; close all;

% Read and Display an Image
originalImage = imread("ImaginiEcografice/imagineTest8.bmp"); % citim imaginea ecografica originala
figure(1), imshow(originalImage), title('Original image to analyze'); % o afisam intr-o figura
hold off

% Extract a bounding box with desired shape to analyze
extractedImage = bwareafilt(originalImage > 0, 1, [1 1 1; 0 1 0; 1 1 1]);
propsExtractedImage = regionprops(extractedImage, 'BoundingBox');
boundingBox = propsExtractedImage.BoundingBox; % punem intr-un Bounding Box ce am extras de pe imaginea ecografica

% Crop original image
croppedImage = imcrop(originalImage, boundingBox); % cropam bounding box-ul din imaginea ecografica originala
figure(2), imshow(croppedImage), title("Cropped Image from original one"); % afisam rezultatul din bounding box

% Resizing cropped image after extract the bounding box and eliminating the noise
[sizeRow, sizeColumn] = size(croppedImage); % imaginea cropata are rezolutia 399x564 
opResize = imresize(croppedImage, [100 100]); % schimbam rezolutia si eliminam noise-ul de pe imagine
resizedImage = imresize(opResize, [sizeRow, sizeColumn]); % readucem la loc imaginea la rezolutia initiala 399x564, fara noise
figure(3), imshow(resizedImage), title('Resized Image without noise'); % afisam imaginea obtinuta dupa eliminare noise

% K means algorithm
numberColors = 4;
[clustedImageWithKmeans, clusterCentroids] = ...
    imsegkmeans(resizedImage, numberColors,'MaxIterations', 100); % aplicam K-Means algorithm 
clusterCentroids = im2double(clusterCentroids); % centroidele fiecarui cluster with K-means (avem 4 in total)

appliedClustedImage = labeloverlay(resizedImage, clustedImageWithKmeans); % adaugam clusterele peste imaginea cropata
figure(4), imshow(appliedClustedImage),
title(['Clustered image with ', num2str(numberColors), ' numColors']); % afisam imaginea clusterizata

clusteredFinalImage = rgb2gray(appliedClustedImage); % transformam imaginea 3d in 2d pentru a o utiliza mai departe
figure(5), imshow(clusteredFinalImage),
title('Final Clustered Image converted in 2D');

doubleClusteredImage = im2double(clusteredFinalImage); % Transform clusteredFinalImage in double 

clusteredImage = doubleClusteredImage;
figure(6), imshow(clusteredImage), title('Original Analyzed Clustered Image converted in double');

% ! Erosion and dilation algorithm %!
graythresh(clusteredImage)

thresholdLevel = graythresh(clusteredImage); % Percent of threshold level between 30-40 %

% Image binarization with threshold level
binarizedClusteredImage = imbinarize(clusteredImage, thresholdLevel); % binarizare imagine cu clustere in functie de threshold-ul sau
figure(9), imshow(~binarizedClusteredImage), title('Binarized image searching for white contour');

% Erode images
% Erode vertical lines
lengthSE = 2; 
SE = ones(3, lengthSE); % define structuring element
bwErodeVL = imerode(~binarizedClusteredImage, SE);   % Erode linii verticale 
subplot(2,2,1), imshow(bwErodeVL),
title('Binarized Image after Eroding Vertical Lines'); % afisam rezultatul

% Erode horizontal lines
bwErodeHL = imerode(bwErodeVL, SE'); % Eroziune linii orizontale 
subplot(2,2,2), imshow(bwErodeHL),
title('Binarized Image after Eroding Vertical and Horizontal Lines');

% Dilate images
% Dilate horizontal lines
bwDilateHL = imdilate(bwErodeHL, SE'); % Dilatare linii orizontale
subplot(2,2,3), imshow(bwDilateHL); 
title('Binarized Image after Dilation on Horizontal Lines');

% Dilate vertical lines
bwFinalResult = imdilate(bwDilateHL, SE);  % Dilatare linii verticale
subplot(2,2,4),
imshow(bwFinalResult),
title('Final resulting image after eroding and dilating methods on binary image'); % afisare rezultate dupa eroziuni si dilatari ale imaginii

% ! Bw Perimeter Algorithm !%
% Find connected components in binary image
connectedComponentsOfResult = bwconncomp(bwFinalResult); % cautam componentele conectate intre ele in imaginea binarizata
regionStatusOfResult = regionprops(connectedComponentsOfResult); % Extragem regiunile de interes pentru fiecare forma de pe imaginea binarizata anterior

% Perimeter of boundaries
perimeterOfBoundaries = bwperim(bwFinalResult); % Identificam perimetrul obiectelor din imaginea binarizata

% Locate the boundaries
[xPointPerimeter, yPointPerimeter] = find(perimeterOfBoundaries); % Marginile perimetrului
boundariesPointsOfPerimeter = [xPointPerimeter, yPointPerimeter];

figure(11), imshow(croppedImage); 
hold on; plot(yPointPerimeter, xPointPerimeter, 'g.'),% afisare detectarea perimetrelor obiectelor gasite
title('Detected contour of perimeters of shapes'), hold off;

% Detected contour overlaying on Cropped Image
detectedContourOnCroppedImage = labeloverlay(croppedImage, bwFinalResult);
figure(12), imshow(detectedContourOnCroppedImage), title('Detected contour overlaying on Cropped Image'); % suprapunem perimetrele gasite peste imaginea cropata

% Extract objects from binary image by descending size
extractedObjectFromBwFinalResult = bwareafilt(bwFinalResult, 4);
figure(13), imshow(extractedObjectFromBwFinalResult), title('Extracted objects from binary image by size');

% Display detected contour after eliminating small object on Cropped Image
extractedOverlayedResult = labeloverlay(croppedImage, extractedObjectFromBwFinalResult);
figure(14), imshow(extractedOverlayedResult), title('Detected contour after eliminating small object');

% Label connected components in 2-D binary image
labelOfBWResult = bwlabel(extractedObjectFromBwFinalResult); % Label connected components in 2-D binary image

% Detecting white shapes
bwResult = bwFinalResult; % copy of binary image bwFinalResult
bwResult(labelOfBWResult~=2 & labelOfBWResult~=3 & labelOfBWResult~=4) = 0;
% put '0' black in points of clusters 2,3,4  from labelOfBWResult

% Display extracted shapes
figure(15), imshow(bwResult), title('Extracting desired shape');

% Filling all holes of shapes
bwResult = imfill(bwResult, 'holes'); %bwResult after filling all holes
figure(16), imshow(bwResult), title('Extracting desired shape with filling all holes')

[exteriorBoundaryOfObject, labelMatrixOfObjects] = bwboundaries(bwResult);