% 0 is black and 1 is white!

clearvars; clc; close all;

% Read and Display an Image
originalImage = imread("imagineTest1.bmp"); % read original_image
figure(1), imshow(originalImage), title('Original image to analyze'); % display original_image
hold off