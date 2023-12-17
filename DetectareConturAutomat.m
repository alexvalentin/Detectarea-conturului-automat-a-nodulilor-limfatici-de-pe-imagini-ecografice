% 0 is black and 1 is white!
% imagine 1 3 8 de test
clearvars; clc; close all;

% Read and Display an Image
originalImage = imread("ImaginiEcografice/imagineTest3.bmp"); % citim imaginea ecografica originala
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
    imsegkmeans(resizedImage, numberColors,'MaxIterations', 100);
%
% aplicam K-Means algorithm 
clusterCentroids = im2double(clusterCentroids) % centroidele fiecarui cluster with K-means (avem 4 in total)

appliedClustedImage = labeloverlay(resizedImage, clustedImageWithKmeans); % adaugam clusterele peste imaginea cropata
figure(4), imshow(appliedClustedImage),
title(['Clustered image with ', num2str(numberColors), ' numColors']); % afisam imaginea clusterizata

clusteredFinalImage = rgb2gray(appliedClustedImage); % transformam volum 3d (spatiu de culori) in 2d pentru a o utiliza mai departe
figure(5), imshow(clusteredFinalImage),
title('Final Clustered Image converted in 2D');

doubleClusteredImage = im2double(clusteredFinalImage); % Transform clusteredFinalImage in double 

clusteredImage = doubleClusteredImage;
figure(6), imshow(clusteredImage), title('Original Analyzed Clustered Image converted in double');


% Image binarization with threshold level
binarizedClusteredImage = imbinarize(clusteredImage); % binarizare imagine cu clustere in functie de threshold-ul sau
figure(9), imshow(~binarizedClusteredImage), title('Binarized image searching for white contour');

bwFinalResult = ~binarizedClusteredImage
% ! Bw Perimeter Algorithm !%
% Find connected components in binary image
connectedComponentsOfResult = bwconncomp(bwFinalResult); % cautam componentele conectate intre ele in imaginea binarizata
regionStatusOfResult = regionprops(connectedComponentsOfResult); % identificam regiunile de interes pentru fiecare forma de pe imaginea binarizata anterior

% Perimeter of boundaries
perimeterOfBoundaries = bwperim(bwFinalResult); % Identificam perimetrul obiectelor din imaginea binarizata

% Locate the boundaries
[xPointPerimeter, yPointPerimeter] = find(perimeterOfBoundaries); % Marginile perimetrului
boundariesPointsOfPerimeter = [xPointPerimeter, yPointPerimeter];

figure(11), imshow(croppedImage); 
hold on; plot(yPointPerimeter, xPointPerimeter, 'g.'),% afisare detectarea perimetrelor obiectelor gasite
title('Detected contour of perimeters of shapes'), hold off;
%
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
% put '0' black in points of clusters 2,3,4 from labelOfBWResult

% Display extracted shapes
figure(15), imshow(bwResult), title('Extracting desired shape');

% Filling all holes of shapes
bwResult = imfill(bwResult, 'holes'); %bwResult after filling all holes
figure(16), imshow(bwResult), title('Extracting desired shape with filling all holes')

% ! Identifying Round Objects Algorithm !%
% Trace region boundaries in binary image
[exteriorBoundaryOfObject, labelMatrixOfObjects] = bwboundaries(bwResult);

% Display the label matrix and draw each boundary.
figure(17), imshow(label2rgb(labelMatrixOfObjects, @jet, [.5 .5 .5])); 
title('Metrics Closer to 1 Indicate that the Object is Approximately Round');
hold on

% Extract exterior of boundaries of each object and plot contour of each
% perimeter
for k = 1 : length(exteriorBoundaryOfObject)
  perimeterOfBoundaries = exteriorBoundaryOfObject{k};
  plot(perimeterOfBoundaries(:,2), perimeterOfBoundaries(:,1), 'w', 'LineWidth', 2);
end

% Status about label Matrix of Objects
statsOfLabelMatrixOfObjects = regionprops(labelMatrixOfObjects, 'Area', 'Centroid', 'Orientation');

% Determine which Objects are Round
metricVector = zeros(3, 1);
areaVector = zeros(3, 1);

% loop over the boundaries
for k = 1 : length(exteriorBoundaryOfObject)

  % obtain (X,Y) boundary coordinates corresponding to label 'k'
  perimeterOfBoundaries = exteriorBoundaryOfObject{k};

  % compute a simple estimate of the object's perimeter
  deltaSq = diff(perimeterOfBoundaries).^2;    
  perimeter = sum(sqrt(sum(deltaSq,2)));
  
  % obtain the area calculation corresponding to label 'k'
  area = statsOfLabelMatrixOfObjects(k).Area
  areaVector(k) = area;
  
  % compute the roundness metric
  metric = 4*pi*area/perimeter^2;
  
  % display the results
  metricString = sprintf('%2.2f', metric);
  metricVector(k) = metric;
  
  % Display the percentage of roundness of each contour
  text(perimeterOfBoundaries(1,2)-35, perimeterOfBoundaries(1,1)+14, metricString, 'Color', 'y', ...
      'FontSize',14,'FontWeight','bold');
   
   if (metric <= 0.06)
      bwResult(labelMatrixOfObjects == k) = 0;
      areaVector(k) = 0;
      figure(18), imshow(bwResult);
   end
   
   if (metric >= 0.9)
      bwResult(labelMatrixOfObjects==k) = 0;
      areaVector(k)=0;
      figure(19), imshow(bwResult);
   end
   
   if (area == max(areaVector))
     bwAreaFinalShape = bwareafilt(bwResult, 2);
     figure(20), imshow(bwAreaFinalShape);
   end
end

% Extracted desired contour
bwAreaFinalShape; % final evidence of desired island of pixels
figure(21), imshow(bwAreaFinalShape), hold on,
boundariesOfExtractedShape = bwboundaries(bwAreaFinalShape, 'noholes');

numberOfBoundaries = size(boundariesOfExtractedShape, 1);
minDistance = zeros(1); % initialise minDistance

%
% ! Minimum distance between each pair of boundaries Algorithm !%
% Find minimum distance between each pair of boundaries
for bnd1 = 1 : numberOfBoundaries
	for bnd2 = 1 : numberOfBoundaries
		if bnd1 == bnd2
			% Can't find distance between the region and itself
			continue;
		end
		boundary1 = boundariesOfExtractedShape{bnd1};
		boundary2 = boundariesOfExtractedShape{bnd2};
        
        overallMinDistance = inf; % Initialize.
		boundary1x = boundary1(:, 2);
		boundary1y = boundary1(:, 1);
		x1=1;   y1=x1;
		x2=1; 	y2=x2;
		
		% For every point in boundary 2, find the distance to every point in boundary 1.
		for k = 1 : size(boundary2, 1)
			% Pick the next point on boundary 2.
			boundary2x = boundary2(k, 2);
			boundary2y = boundary2(k, 1);
			% For this point, compute distances from it to all points in boundary 1.
			allDistances = sqrt((boundary1x - boundary2x).^2 + (boundary1y - boundary2y).^2);
			% Find closest point, min distance.
			[minDistance(k), indexOfMin] = min(allDistances);
			if (minDistance(k) < overallMinDistance)
				x1 = boundary1x(indexOfMin);
				y1 = boundary1y(indexOfMin);
				x2 = boundary2x;
				y2 = boundary2y;
				overallMinDistance = minDistance(k);
			end
		end
		% Find the overall min distance
		minDistance = min(minDistance);
        minDistanceString = sprintf('%2.2f', minDistance);
        
		% Report to command window.
		%fprintf('Minimum distance from area %d to area %d is %.3f pixels.\n', bnd1, bnd2, minDistance);

		% Draw a line between point 1 and 2
		line([x1, x2], [y1, y2], 'Color', 'y', 'LineWidth', 3);
       
	end
end

text(x2, y2-20, minDistanceString, 'Color', 'g', ...
      'FontSize',14,'FontWeight','bold');
title('Calculating distance between regions');
hold off;
%%
% Verify existence of bwAreaFinalShape
% Mini algorithm to keep the closest area and remove long distance regions
existBwFinalShape = exist('bwAreaFinalShape', 'var');
if (existBwFinalShape == 1)
    if (minDistance <= 27)
        bwAreaFinalShape = bwareafilt(bwAreaFinalShape, 2);
        figure(22), imshow(bwAreaFinalShape);
    else
        bwAreaFinalShape = bwareafilt(bwAreaFinalShape, 1);
        figure(22), imshow(bwAreaFinalShape);
    end
end
%
% Extract overlay final shape
extractedOverlayedFinalShape = labeloverlay(croppedImage, bwAreaFinalShape); % Overlay label matrix regions on a 2-D image
figure(23), imshow(extractedOverlayedFinalShape), title('Final automatic detected contour on Cropped Image');

% Compare cropped image with final automatic detected contour result
figure(24), montage({croppedImage, extractedOverlayedFinalShape}), title('Original Cropped Image vs. Overlayed Image with final automatic contour');
%
% ! Activecontour Algorithm !%
% Set the mask for activecontour Algorithm !%
maskOfShape = bwAreaFinalShape; % extracted mask

% The extracted mask as automated contour
figure(25), imshow(maskOfShape); % Display the segmented image with activecontour
hold on;
visboundaries(maskOfShape, 'Color', 'y'); % Display the final contour over the original image in yellow
