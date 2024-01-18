% 0 is black and 1 is white!
% imagine  3 6 8 9 11 12 13, 5, 7 de test
clearvars; clc; close all;

% Read and Display an Image
originalImage = imread("ImaginiEcografice/imagineTest1.bmp"); % citim imaginea ecografica originala
figure(1), imshow(originalImage), title('Original image to analyze'); % o afisam intr-o figura
hold off
%
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
clusterCentroids = im2double(clusterCentroids); % centroidele fiecarui cluster with K-means (avem 4 in total)

appliedClustedImage = labeloverlay(resizedImage, clustedImageWithKmeans); % adaugam clusterele peste imaginea cropata
figure(4), imshow(appliedClustedImage),
title(['Clustered image with ', num2str(numberColors), ' numColors']); % afisam imaginea clusterizata

%%% Pentru ACTIVE CONTUR imaginea de segmentat
clusteredFinalImage = rgb2gray(appliedClustedImage); % transformam volum 3d (spatiu de culori) in 2d pentru a o utiliza mai departe
figure(5), imshow(clusteredFinalImage),
title('Final Clustered Image converted in 2D');

doubleClusteredImage = im2double(clusteredFinalImage); % Transform clusteredFinalImage in double 

clusteredImage = doubleClusteredImage;
figure(6), imshow(clusteredImage), title('Original Analyzed Clustered Image converted in double');


% Image binarization with threshold level
binarizedClusteredImage = imbinarize(clusteredImage); % binarizare imagine cu clustere in functie de threshold-ul sau
figure(9), imshow(~binarizedClusteredImage), title('Binarized image searching for white contour');

% Erode images
% Erode vertical lines
lengthSE = 2; 
SE = ones(3, lengthSE);
bwErodeVL = imerode(~binarizedClusteredImage, SE); 
% Erode horizontal lines
bwErodeHL = imerode(bwErodeVL, SE'); 
% Dilate horizontal lines
bwDilateHL = imdilate(bwErodeHL, SE'); 
% Dilate vertical lines
bwFinalResult = imdilate(bwDilateHL, SE);


%bwFinalResult = ~binarizedClusteredImage;
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
%
% Display the label matrix and draw each boundary.
figure(17), imshow(label2rgb(labelMatrixOfObjects, @jet, [.5 .5 .5])); 
title('Metrics Closer to 1 Indicate that the Object is Approximately Round');
hold on

% Extract exterior of boundaries of each object and plot contour of each perimeter
% Parcurgem fiecare pixel de pe marginea exterioara a fiecarei grupari de  pixeli din imagine si plotam ca sa apara conturului fiecare grupari de
% pixeli care formeaza un obiect
for k = 1 : length(exteriorBoundaryOfObject)
  perimeterOfBoundaries = exteriorBoundaryOfObject{k};
  plot(perimeterOfBoundaries(:,2), perimeterOfBoundaries(:,1), 'w', 'LineWidth', 2);
end

% Status about label Matrix of Objects
statsOfLabelMatrixOfObjects = regionprops(labelMatrixOfObjects, 'Area', 'Centroid', 'Orientation'); % calculam aria fiecarui obiect

% Determine which Objects are Round
metricVector = zeros(3, 1);
areaVector = zeros(3, 1);

% loop over the boundaries
for k = 1 : length(exteriorBoundaryOfObject)

  % obtain (X,Y) boundary coordinates corresponding to label 'k'
  perimeterOfBoundaries = exteriorBoundaryOfObject{k}; % coordonatele perimetrului fiecarui obiect

  % compute a simple estimate of the object's perimeter
  deltaSq = diff(perimeterOfBoundaries).^2;   %calculam delta care e diferenta dintre pixelii adiacenti
  perimeter = sum(sqrt(sum(deltaSq,2)));  % perimetru fiecarui obiect
  
  % obtain the area calculation corresponding to label 'k'
  area = statsOfLabelMatrixOfObjects(k).Area
  areaVector(k) = area; % afisam aria
  
  % compute the roundness metric
  metric = 4*pi*area/perimeter^2;  % metrica = 4 pi * area / (perimeter^2)
  
  % display the results
  metricString = sprintf('%2.2f', metric);
  metricVector(k) = metric;
  
  % Display the percentage of roundness of each contour
  text(perimeterOfBoundaries(1,2)-35, perimeterOfBoundaries(1,1)+14, metricString, 'Color', 'y', ...
      'FontSize',14,'FontWeight','bold');
   
   if (metric <= 0.07)
      bwResult(labelMatrixOfObjects == k) = 0;   % pentru metrici foarte mici sau mari, le eliminam
      areaVector(k) = 0;
      figure(18), imshow(bwResult);
   end
   
   if (metric >= 0.9)
      bwResult(labelMatrixOfObjects==k) = 0;
      areaVector(k)=0;
      figure(19), imshow(bwResult);
   end

   if (area >= 40000)
      bwResult(labelMatrixOfObjects==k) = 0;
      areaVector(k)=0;
      bwAreaFinalShape = bwareafilt(bwResult, 1);
      figure(20), imshow(bwAreaFinalShape);
   end
   
   if (area == max(areaVector))                    % dupa ce eliminam fiecare metrica mare sau mica, pastram pe cea cu cea mai mare arie.
     bwAreaFinalShape = bwareafilt(bwResult, 2);
     figure(20), imshow(bwAreaFinalShape);
   end
   
end


% Extracted desired contour
bwAreaFinalShape; % final evidence of desired island of pixels
figure(21), imshow(bwAreaFinalShape), hold on,
boundariesOfExtractedShape = bwboundaries(bwAreaFinalShape, 'noholes');

numberOfBoundaries = size(boundariesOfExtractedShape, 1);   % avem doar 2 obiecte 
minDistance = zeros(1); % initialise minDistance


%
% ! Minimum distance between each pair of boundaries Algorithm !%
% Find minimum distance between each pair of boundaries

% Calculam distanta minima dintre gruparile de pixeli ramase sa o pastram pe cea centrala
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
			allDistances = sqrt((boundary1x - boundary2x).^2 + (boundary1y - boundary2y).^2); % (Xb-Xa)^2 + (Yb-Ya)^2
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

%text(x2, y2-20, minDistanceString, 'Color', 'g', ...
 %   'FontSize',14,'FontWeight','bold');
title('Calculating distance between regions');
hold off;
%
% Verify existence of bwAreaFinalShape
% Mini algorithm to keep the closest area and remove long distance regions
existBwFinalShape = exist('bwAreaFinalShape', 'var');
if (existBwFinalShape == 1)
    if (minDistance <= 27 || (minDistance >= 54.73 && minDistance <=54.75))
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

% Finding white pixels from maskOfShape
[rowContourOfMask, columnContourOfMask] = find(maskOfShape == 1); % find all elements ofthe white section from contour
positionOfCoordinatesOfContourOfMask = [rowContourOfMask, columnContourOfMask]; % row = y | col =x
%
% Column 1 = y, column 2 = x
[maxY, indexOfXinMaxY] = max(positionOfCoordinatesOfContourOfMask(:, 1));
xInMaxY = positionOfCoordinatesOfContourOfMask(indexOfXinMaxY, 2);
P_maxY = [xInMaxY, maxY];

[minY, indexOfXinMinY] = min(positionOfCoordinatesOfContourOfMask(:, 1));
xInMinY = positionOfCoordinatesOfContourOfMask(indexOfXinMinY, 2);
P_minY = [xInMinY, minY];

[maxX, indexOfYinMaxX] = max(positionOfCoordinatesOfContourOfMask(:, 2));
yInMaxX = positionOfCoordinatesOfContourOfMask(indexOfYinMaxX, 1);
P_maxX = [maxX, yInMaxX];

[minX, indexOfYinMinX] = min(positionOfCoordinatesOfContourOfMask(:, 2));
yInMinX = positionOfCoordinatesOfContourOfMask(indexOfYinMinX, 1);
P_minX = [minX, yInMinX];

% Limits of the shapes
plot(P_maxY(1), P_maxY(2), '*m'); 
plot(P_minY(1), P_minY(2), '*g'); 
plot(P_maxX(1), P_maxX(2), '*r'); 
plot(P_minX(1), P_minX(2), '*b'); 

%Centroid of the shape
centroidOfShape = [sum(positionOfCoordinatesOfContourOfMask(:, 2))/length(positionOfCoordinatesOfContourOfMask(:, 2)), sum(positionOfCoordinatesOfContourOfMask(:, 1))/length(positionOfCoordinatesOfContourOfMask(:, 1))];
plot(centroidOfShape(1), centroidOfShape(2), '+c')
title('Limits of coordinates points of contour');
hold off; 

figure(26), imshow(croppedImage), hold on;
visboundaries(maskOfShape, 'Color', 'g'), title('Cropped Image with automated detected contour');
hold off;

figure(27), imshow(maskOfShape), hold on,
title('Contour of the mask with green');
%
% Creating the ellipse
createdEllipse = images.roi.Ellipse(gca, 'Center', centroidOfShape, 'Semiaxes',[(P_maxX(1)-P_minX(1))/2 (P_maxY(2)-P_minY(2))/2]);
maskOfEllipse = createMask(createdEllipse);
contour(maskOfEllipse, 'Color', [0, 0.2, 0.4]) % green countour on mask
hold off;

regionEllipse = regionprops(maskOfEllipse, 'BoundingBox', 'PixelList', 'ConvexImage');
boundingBoxEllipse = regionEllipse.BoundingBox;
sampleEllipse = imcrop(croppedImage, boundingBoxEllipse);

figure(28), imshow(sampleEllipse);

% Intensity of pixels from ellipse
intensityOfEllipsePixels = mean(mean(sampleEllipse));

% Determining the smooth factor for active contour algoritm using intensity of pixels from the ellipse 
if (intensityOfEllipsePixels <= 38)
    smoothFactorForActiveContour = 4;
else
    smoothFactorForActiveContour = 2;
end

% Active Contour Algorithm
imageAfterActiveContour = activecontour(clusteredFinalImage, maskOfShape, 300, 'edge', ...
    'SmoothFactor', smoothFactorForActiveContour, 'ContractionBias', -1);

% Finding white pixels from imageAfterActiveContour
[rowContourAC, columnContourAC] = find(imageAfterActiveContour == 1); % find all elements ofthe white section from contour
positionOfCoordinatesOfContour = [rowContourAC, columnContourAC]; % row = y | col =x

% Limits point on x and y axis for image after snake algorithm
% Column 1 = y, column 2 = x
[maxY, indexOfXinMaxY] = max(positionOfCoordinatesOfContour(:, 1));
xInMaxY = positionOfCoordinatesOfContour(indexOfXinMaxY, 2);
P_maxY = [xInMaxY, maxY];

[minY, indexOfXinMinY] = min(positionOfCoordinatesOfContour(:, 1));
xInMinY = positionOfCoordinatesOfContour(indexOfXinMinY, 2);
P_minY = [xInMinY, minY];

[maxX, indexOfYinMaxX] = max(positionOfCoordinatesOfContour(:, 2));
yInMaxX = positionOfCoordinatesOfContour(indexOfYinMaxX, 1);
P_maxX = [maxX, yInMaxX];

[minX, indexOfYinMinX] = min(positionOfCoordinatesOfContour(:, 2));
yInMinX = positionOfCoordinatesOfContour(indexOfYinMinX, 1);
P_minX = [minX, yInMinX];
%
% The length in mm of each axis that compute the shape
lengthOnYaxis = norm(P_minY - P_maxY) / 12; % 1 segment are 10 mm 48.8 pixeli 
lengthOnXaxis = norm(P_minX - P_maxX) / 12;


figure(29), imshow(croppedImage), hold on; %aici sa intreb
phiContour = bwdist(imageAfterActiveContour) - bwdist(1-imageAfterActiveContour) + im2double(imageAfterActiveContour);
contour(phiContour, [0 0], 'Color', [1, 0.6, 0.25], 'LineWidth', 2); % phi contour in orange

% Limits of the shapes
plot(P_maxY(1), P_maxY(2), '*m'); plot(P_minY(1), P_minY(2), '*g'); 
plot(P_maxX(1), P_maxX(2), '*r'); plot(P_minX(1), P_minX(2), '*b'); 
title('The obtained contour after using Active Contour algorithm');
hold off;

% Display obtained contour and its contour on cropped image 
figure(30), imshow(croppedImage), hold on;
contour(phiContour, [0 0],'Color',[1, 0.6, 0.25],'LineWidth',2);
%
plot(P_maxY(1), P_maxY(2), '*m'); plot(P_minY(1), P_minY(2), '*g');
plot(P_maxX(1), P_maxX(2), '*r'); plot(P_minX(1), P_minX(2), '*b'); 

plot((sum(positionOfCoordinatesOfContour(:, 2)) + minX)/length(positionOfCoordinatesOfContour(:, 2)), ...
    (sum(positionOfCoordinatesOfContour(:, 1)) + minY)/length(positionOfCoordinatesOfContour(:, 1)), '+b')

lineBtX = drawline('Position',[P_minX; P_maxX],'StripeColor','r', 'Label', num2str(lengthOnXaxis), 'LabelTextColor', 'g');
lineBtY = drawline('Position',[P_minY; P_maxY],'StripeColor','g', 'Label', [num2str(lengthOnYaxis), newline, '']);

